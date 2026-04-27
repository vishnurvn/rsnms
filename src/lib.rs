use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{
    Bound, PyResult, Python, pyfunction, pymodule,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction,
};

pub mod ops;
pub mod utils;

#[pyfunction]
fn nms<'py>(
    py: Python<'py>,
    bboxes: PyReadonlyArray2<f32>,
    confs: PyReadonlyArray1<f32>,
    iou_threshold: f32,
) -> Bound<'py, PyArray2<f32>> {
    if !bboxes.is_contiguous() && !confs.is_contiguous() {
        panic!("Boxes are not contigous. Run np.ascontigous before passing it to the function")
    }
    let bboxes_view = bboxes.as_array();
    let confs_view = confs.as_array();

    let confs_vec = confs_view.to_vec();
    let bboxs_vec: Vec<[f32; 4]> = bboxes_view
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2], row[3]])
        .collect();

    let res = ops::non_max_suppression_rs(bboxs_vec, confs_vec, iou_threshold);
    let res_arr = Array2::from(res);
    return res_arr.into_pyarray(py);
}

#[pymodule]
fn rsnms(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nms, m)?)?;
    return Ok(());
}
