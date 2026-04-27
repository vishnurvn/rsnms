use rsnms::{ops::non_max_suppression_rs, utils::calc_iou};
use rstest::rstest;

#[rstest]
#[case([0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], 1.0)]
#[case([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0], 0.0)]
#[case([0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0], 1.0 / 7.0)]
#[case([1.0, 0.0, 3.0, 2.0], [0.0, 1.0, 2.0, 3.0], 1.0 / 7.0)]
fn iou_all(#[case] box1: [f32; 4], #[case] box2: [f32; 4], #[case] expected: f32) {
    let iou = calc_iou(&box1, &box2);
    assert_eq!(iou, expected);
}

/// Basic supression test case
#[test]
fn nms_simple() {
    let bboxes = vec![[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]];
    let confs = vec![0.9, 0.8];

    let result = non_max_suppression_rs(bboxes, confs, 0.5);
    assert_eq!(result.len(), 1usize);
}

/// only one box present
#[test]
fn nms_single_box() {
    let bboxes: Vec<[f32; 4]> = Vec::from([[0.0, 0.0, 1.0, 1.0]]);
    let confs: Vec<f32> = Vec::from([0.9]);
    let result = non_max_suppression_rs(bboxes, confs, 0.5);
    assert_eq!(result.len(), 1usize);
}

/// All boxes survive the supression
#[test]
fn nms_all_surviving() {
    let bboxes: Vec<[f32; 4]> = vec![
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 2.0, 2.0],
        [2.0, 2.0, 3.0, 3.0],
    ];
    let confs: Vec<f32> = vec![0.9, 0.9, 0.9];
    let result = non_max_suppression_rs(bboxes, confs, 0.5);
    assert_eq!(result.len(), 3usize);
}

/// Overlap less than the threshold, all boxes survives
#[test]
fn nms_lt_threshold() {
    let bboxes: Vec<[f32; 4]> = vec![[120.0, 80.0, 320.0, 280.0], [200.0, 160.0, 400.0, 360.0]];
    let confs: Vec<f32> = vec![0.9, 0.9];
    let result = non_max_suppression_rs(bboxes, confs, 0.5);
    assert_eq!(result.len(), 2usize);
}

/// no boxes at all
#[test]
fn nms_no_boxes() {
    let bboxes: Vec<[f32; 4]> = Vec::new();
    let confs: Vec<f32> = Vec::new();
    let result = non_max_suppression_rs(bboxes, confs, 0.5);
    assert_eq!(result.len(), 0usize);
}
