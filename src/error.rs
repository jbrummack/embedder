#[cfg(feature = "burn")]
use burn::tensor::DataError;
use image::ImageError;
//use qdrant_client::QdrantError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbedderError {
    #[cfg(feature = "burn")]
    #[error("{0:?}")]
    Burn(DataError),
    #[cfg(not(feature = "burn"))]
    #[error("Burn isnt compiled in")]
    Burn,
    #[cfg(feature = "ort")]
    #[error("{0:?}")]
    Ort(#[from] ort::Error),
    #[cfg(not(feature = "ort"))]
    #[error("Compiled without ONNX RT")]
    Ort,
    #[error("{0}")]
    Image(#[from] ImageError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

pub type EmbedderResult<T> = Result<T, EmbedderError>;
