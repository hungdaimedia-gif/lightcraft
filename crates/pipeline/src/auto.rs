//! Auto tone and auto white balance (histogram / grey-world statistics on a proxy).

use lightcraft_color::cct::xy_to_temp_tint;
use lightcraft_color::transfer::decode_srgb8;
use lightcraft_color::{REC2020, Xy, bradford, luminance_2020};
use lightcraft_develop::DevelopSettings;
use lightcraft_raster::{Rgb32f, Rgba8};
use serde::Serialize;

use crate::SourceInfo;
use crate::local::effective_wb;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct AutoTone {
    pub exposure: f64,
    pub contrast: f64,
    pub highlights: f64,
    pub shadows: f64,
    pub whites: f64,
    pub blacks: f64,
    pub vibrance: f64,
    pub saturation: f64,
}

fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f32 * q.clamp(0.0, 1.0)) as usize]
}

/// Compute auto tone values for `src` under the current white balance (ignores current tone values).
pub fn auto_tone(src: &Rgb32f, info: &SourceInfo, s: &DevelopSettings) -> AutoTone {
    let mut img = lightcraft_raster::resample::fit(src, 512, 512, lightcraft_raster::resample::Filter::Box);
    let mut base = DevelopSettings { wb: s.wb, ..DevelopSettings::default() };
    base.light.exposure = 0.0;
    crate::local::scene_linear_pre(&mut img, info, &base);
    let mut ev: Vec<f32> = img.data.iter().map(|c| (luminance_2020(*c).max(1e-6) / 0.18).log2()).collect();
    ev.sort_by(|a, b| a.total_cmp(b));
    let median = percentile(&ev, 0.5);
    let exposure = (-median * 0.85 - 0.1).clamp(-4.0, 4.0);
    let (p01, p05, p95, p995) =
        (percentile(&ev, 0.01) + exposure, percentile(&ev, 0.05) + exposure, percentile(&ev, 0.95) + exposure, percentile(&ev, 0.995) + exposure);
    let highlights = if p995 > 2.2 { -((p995 - 2.2) * 38.0).min(90.0) } else { 0.0 };
    let shadows = if p05 < -4.0 { ((-4.0 - p05) * 22.0).min(70.0) } else { 0.0 };
    let spread = p95 - p05;
    let contrast = ((6.5 - spread) * 6.0).clamp(-20.0, 30.0);
    let whites = if p995 < 1.8 { ((1.8 - p995) * 25.0).min(40.0) } else { -((p995 - 3.5).max(0.0) * 10.0).min(30.0) };
    let blacks = if p01 > -5.0 { -((p01 + 5.0) * 10.0).min(35.0) } else { ((-7.0 - p01).max(0.0) * 8.0).min(20.0) };
    AutoTone {
        exposure: (exposure as f64 * 100.0).round() / 100.0,
        contrast: contrast.round() as f64,
        highlights: highlights.round() as f64,
        shadows: shadows.round() as f64,
        whites: whites.round() as f64,
        blacks: blacks.round() as f64,
        vibrance: 12.0,
        saturation: 3.0,
    }
}

/// An automatic black & white mix (slider values, red … magenta) for `src` under `s`'s white
/// balance and tone: each hue band's colourful pixels are pushed away from the image's mean
/// lightness — bands brighter than average get brighter, darker ones darker — so areas that
/// differ only in colour stay apart in grey. Bands with almost no colourful pixels stay at 0.
pub fn auto_bw_mix(src: &Rgb32f, info: &SourceInfo, s: &DevelopSettings) -> [f64; 8] {
    use lightcraft_color::perceptual::{lab_to_lch, oklab_from_2020};
    let mut img = lightcraft_raster::resample::fit(src, 512, 512, lightcraft_raster::resample::Filter::Box);
    let base = DevelopSettings { wb: s.wb, light: s.light, ..DevelopSettings::default() };
    crate::local::scene_linear_pre(&mut img, info, &base);
    let gain = 2f32.powf(base.light.exposure as f32);
    let (mut mass, mut sum_l) = ([0f64; 8], [0f64; 8]);
    let (mut all_l, mut n) = (0f64, 0usize);
    for p in &img.data {
        let lch = lab_to_lch(oklab_from_2020(p.map(|v| (v * gain).max(0.0))));
        all_l += lch[0] as f64;
        n += 1;
        let k = (lch[1] / 0.2).min(1.0) as f64; // as in the B&W conversion
        if k < 0.05 {
            continue;
        }
        let w = crate::colorops::band_weights(lch[2]);
        for i in 0..8 {
            mass[i] += w[i] as f64 * k;
            sum_l[i] += w[i] as f64 * k * lch[0] as f64;
        }
    }
    if n == 0 {
        return [0.0; 8];
    }
    let mean = all_l / n as f64;
    let total: f64 = mass.iter().sum();
    std::array::from_fn(|i| {
        if total <= 0.0 || mass[i] / total < 0.01 {
            return 0.0;
        }
        let sep = sum_l[i] / mass[i] - mean;
        (sep * 400.0).clamp(-60.0, 60.0).round()
    })
}

/// Grey-world white balance weighted towards mid-tone, low-chroma pixels. Returns (temp, tint).
pub fn auto_wb(src: &Rgb32f, info: &SourceInfo) -> (f64, f64) {
    let img = lightcraft_raster::resample::fit(src, 256, 256, lightcraft_raster::resample::Filter::Box);
    let (mut acc, mut wsum) = ([0.0f64; 3], 0.0f64);
    for c in &img.data {
        let y = luminance_2020(*c);
        if !(0.01..=2.0).contains(&y) {
            continue;
        }
        let mx = c[0].max(c[1]).max(c[2]);
        let mn = c[0].min(c[1]).min(c[2]);
        let chroma = (mx - mn) / (mx + 1e-6);
        let w = (1.0 - chroma).powi(2) as f64 * (1.0 - ((y.log2() + 2.5) / 4.0).abs().min(1.0)) as f64;
        for i in 0..3 {
            acc[i] += c[i] as f64 * w;
        }
        wsum += w;
    }
    if wsum <= 0.0 {
        return (info.as_shot_temp, info.as_shot_tint);
    }
    let avg = acc.map(|v| v / wsum);
    let xyz = REC2020.to_xyz().apply(avg);
    let shot = lightcraft_color::cct::temp_tint_to_xy(info.as_shot_temp, info.as_shot_tint);
    let seen = bradford(REC2020.white, shot).apply(xyz);
    let (t, tint) = xy_to_temp_tint(Xy::from_xyz(seen));
    (t.clamp(2000.0, 50000.0).round(), tint.clamp(-150.0, 150.0).round())
}

/// Temperature/tint currently in effect (for UI display).
pub fn current_wb(info: &SourceInfo, s: &DevelopSettings) -> (f64, f64) {
    effective_wb(info, s)
}

/// Spread of the centre weighting in [`subject_brightness`], as a fraction of the frame: a subject
/// filling the central half of the frame carries ~64% of the weight, so it, not the background,
/// sets the median.
const SUBJECT_SIGMA: f32 = 0.2;

/// How bright the subject of a *rendered* (display-encoded 8-bit) image looks: the weighted
/// median of log2 linear luminance, weighted towards the centre of the frame. Event and portrait
/// photos put people near the centre, so they decide "too dark / too bright" rather than a
/// window or a dark wall at the edge. `None` for an empty image.
pub fn subject_brightness(img: &Rgba8) -> Option<f64> {
    let (w, h) = (img.width, img.height);
    if w == 0 || h == 0 {
        return None;
    }
    let mut samples: Vec<(f32, f32)> = Vec::with_capacity(img.data.len());
    for (i, px) in img.data.iter().enumerate() {
        let dx = ((i % w) as f32 + 0.5) / w as f32 - 0.5;
        let dy = ((i / w) as f32 + 0.5) / h as f32 - 0.5;
        let weight = (-(dx * dx + dy * dy) / (2.0 * SUBJECT_SIGMA * SUBJECT_SIGMA)).exp();
        let y = 0.2126 * decode_srgb8(px[0]) + 0.7152 * decode_srgb8(px[1]) + 0.0722 * decode_srgb8(px[2]);
        samples.push((y.max(1e-4).log2(), weight));
    }
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f32 = samples.iter().map(|s| s.1).sum();
    if total.is_nan() || total <= 0.0 {
        return None;
    }
    let mut acc = 0.0f32;
    for (ev, weight) in &samples {
        acc += weight;
        if acc >= total * 0.5 {
            return Some(*ev as f64);
        }
    }
    samples.last().map(|s| s.0 as f64)
}

/// Close enough: 0.03 EV is well below what anyone sees between two photos in a set.
const BALANCE_TOLERANCE: f64 = 0.03;

/// The Exposure slider value at which `measure` (render with that Exposure, then
/// [`subject_brightness`]) reaches `target`, starting from `start`. Exposure moves log
/// luminance almost one-for-one before the tone curve, so a few secant steps converge; the
/// result stays in the slider's −5..5 range. `None` when the image can't be measured.
pub fn solve_exposure(start: f64, target: f64, mut measure: impl FnMut(f64) -> Option<f64>) -> Option<f64> {
    let (mut x0, mut y0) = (start, measure(start)? - target);
    if y0.abs() < BALANCE_TOLERANCE {
        return Some(x0);
    }
    let mut x1 = (x0 - y0).clamp(-5.0, 5.0);
    for _ in 0..6 {
        let y1 = measure(x1)? - target;
        if y1.abs() < BALANCE_TOLERANCE {
            return Some(x1);
        }
        let slope = (y1 - y0) / (x1 - x0);
        // a flat or broken slope (clipped whites, black frame) falls back to a plain one-to-one step
        let next = if slope.is_finite() && slope.abs() > 0.05 { x1 - y1 / slope } else { x1 - y1 };
        (x0, y0) = (x1, y1);
        x1 = next.clamp(-5.0, 5.0);
        if (x1 - x0).abs() < 1e-3 {
            break;
        }
    }
    Some(x1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_image_gets_positive_exposure() {
        let img = Rgb32f::from_fn(64, 64, |x, _| [0.01 + x as f32 * 0.0003; 3]);
        let a = auto_tone(&img, &SourceInfo::default(), &DevelopSettings::default());
        assert!(a.exposure > 1.5, "{a:?}");
        let bright = Rgb32f::from_fn(64, 64, |x, _| [0.8 + x as f32 * 0.01; 3]);
        let b = auto_tone(&bright, &SourceInfo::default(), &DevelopSettings::default());
        assert!(b.exposure < -1.0, "{b:?}");
    }

    #[test]
    fn neutral_image_keeps_as_shot_wb() {
        let img = Rgb32f::from_fn(32, 32, |x, y| [0.05 + (x + y) as f32 * 0.004; 3]);
        let (t, tint) = auto_wb(&img, &SourceInfo::default());
        assert!((t - 6500.0).abs() < 150.0, "{t}");
        assert!(tint.abs() < 6.0, "{tint}");
    }

    #[test]
    fn blue_cast_is_corrected_by_higher_temp() {
        // A bluish cast should be neutralised by telling the pipeline the light was bluer (higher K).
        let img = Rgb32f::from_fn(32, 32, |_, _| [0.16, 0.18, 0.24]);
        let (t, _) = auto_wb(&img, &SourceInfo::default());
        assert!(t > 7000.0, "{t}");
        let mut s = DevelopSettings::default();
        s.wb.mode = lightcraft_develop::WbMode::Custom;
        s.wb.temp = t;
        let (t2, tint2) = auto_wb(&img, &SourceInfo::default());
        let _ = (t2, tint2);
        let mut out = img.clone();
        let (tt, ti) = auto_wb(&img, &SourceInfo::default());
        s.wb.temp = tt;
        s.wb.tint = ti;
        crate::local::scene_linear_pre(&mut out, &SourceInfo::default(), &s);
        let c = out.get(0, 0);
        assert!((c[0] - c[2]).abs() < 0.02, "{c:?}");
    }

    #[test]
    fn subject_brightness_follows_the_centre() {
        // a subject filling the central half, bright on dark vs. the reverse: the subject decides
        let frame = |centre: u8, edge: u8| {
            let mut img = Rgba8::new(64, 64);
            for (i, px) in img.data.iter_mut().enumerate() {
                let (x, y) = (i % 64, i / 64);
                let v = if (16..48).contains(&x) && (16..48).contains(&y) { centre } else { edge };
                *px = [v, v, v, 255];
            }
            img
        };
        let lit = subject_brightness(&frame(200, 30)).unwrap();
        let dark = subject_brightness(&frame(30, 200)).unwrap();
        assert!(lit > dark + 2.0, "{lit} vs {dark}");
        assert_eq!(subject_brightness(&Rgba8::new(0, 0)), None);
    }

    #[test]
    fn solve_exposure_hits_the_target() {
        // an idealised pipeline: brightness = exposure − 1.3, but whites clip at +1
        let measure = |e: f64| Some((e - 1.3).min(1.0));
        let e = solve_exposure(0.0, -0.4, measure).unwrap();
        assert!((e - 0.9).abs() < BALANCE_TOLERANCE, "{e}");
        // a compressive tone curve (half slope) still converges
        let soft = |e: f64| Some(0.5 * e - 1.0);
        let e = solve_exposure(0.0, -0.5, soft).unwrap();
        assert!((e - 1.0).abs() < 0.1, "{e}");
        // unmeasurable images give up instead of guessing
        assert_eq!(solve_exposure(0.0, 0.0, |_| None), None);
        // never leaves the slider range
        let e = solve_exposure(0.0, 50.0, Some).unwrap();
        assert!(e <= 5.0, "{e}");
    }
}

#[cfg(test)]
mod bw_tests {
    use super::*;

    #[test]
    fn auto_bw_mix_pushes_light_and_dark_hues_apart() {
        // left: a light yellow, right: a dark blue (scene-linear Rec. 2020)
        let src = Rgb32f::from_fn(64, 32, |x, _| if x < 32 { [0.55, 0.5, 0.05] } else { [0.01, 0.02, 0.12] });
        let m = auto_bw_mix(&src, &SourceInfo::default(), &DevelopSettings::default());
        let (yellow, blue) = (m[2], m[5]);
        assert!(yellow > 0.0 && blue < 0.0, "{m:?}");
        // an image without colour leaves the mix alone
        let grey = Rgb32f::from_fn(16, 16, |x, _| [x as f32 / 16.0; 3]);
        assert_eq!(auto_bw_mix(&grey, &SourceInfo::default(), &DevelopSettings::default()), [0.0; 8]);
    }
}
