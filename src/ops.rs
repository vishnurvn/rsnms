use std::iter::zip;

use crate::utils::calc_iou;

pub fn non_max_suppression_rs(
    bboxes: Vec<[f32; 4]>,
    confs: Vec<f32>,
    iou_threshold: f32,
) -> Vec<[f32; 4]> {
    assert_eq!(bboxes.len(), confs.len());
    let mut order: Vec<usize> = (0..confs.len()).collect();
    let mut mask: Vec<bool> = vec![true; order.len()];
    order.sort_by(|&a, &b| confs[b].total_cmp(&confs[a]));

    if order.is_empty() {
        return Vec::new();
    }

    mask[order[0]] = false;

    for (skip, major_idx) in order.iter().enumerate() {
        let master_bbox = bboxes[*major_idx];
        for minor_idx in order.iter().skip(skip + 1) {
            if !mask[*minor_idx] {
                continue;
            }
            let to_cmp_bbox = bboxes[*minor_idx];
            let iou = calc_iou(&master_bbox, &to_cmp_bbox);
            if iou < iou_threshold {
                mask[*minor_idx] = false;
            }
        }
    }

    let mut final_bboxes: Vec<[f32; 4]> = Vec::new();
    println!("{:?}", final_bboxes);
    for (bbox, m) in zip(bboxes, mask) {
        if !m {
            final_bboxes.push(bbox);
        }
    }
    return final_bboxes;
}
