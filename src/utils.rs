/// Computes the iou of 2 boxes
pub fn calc_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let [x1_min, y1_min, x1_max, y1_max] = *box1;
    let [x2_min, y2_min, x2_max, y2_max] = *box2;

    // Intersection
    let inter_x_min = x1_min.max(x2_min);
    let inter_y_min = y1_min.max(y2_min);
    let inter_x_max = x1_max.min(x2_max);
    let inter_y_max = y1_max.min(y2_max);

    let inter_w = (inter_x_max - inter_x_min).max(0.0);
    let inter_h = (inter_y_max - inter_y_min).max(0.0);
    let inter_area = inter_w * inter_h;

    if inter_area == 0.0 {
        return 0.0;
    }

    // Union
    let area1 = (x1_max - x1_min) * (y1_max - y1_min);
    let area2 = (x2_max - x2_min) * (y2_max - y2_min);
    let union_area = area1 + area2 - inter_area;

    inter_area / union_area
}
