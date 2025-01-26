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
    #[error("{0:?}")]
    Ort(#[from] ort::Error),
    #[error("{0}")]
    Image(#[from] ImageError),
    //#[error("{0}")]
    //VecDB(#[from] QdrantError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

pub type EmbedderResult<T> = Result<T, EmbedderError>;
