//uniffi::include_scaffolding!("embedder");

use std::sync::OnceLock;

use burn_ndarray::NdArray;

use crate::{error::EmbedderResult, EfficientNetEmbedder};

pub static EMBEDDER: OnceLock<EfficientNetEmbedder<NdArray>> = OnceLock::new();
//pub static EFFICIENTNET_WEIGHTS: &[u8] = include_bytes!("model/efficientnet.bin");

pub fn embed(request: EmbeddingRequest) -> EmbedderResult<Embedding> {
    let embedder = EMBEDDER.get_or_init(|| EfficientNetEmbedder::new().unwrap());
    println!("{request:#?}");
    let EmbeddingRequest {
        image_data,
        backend: _,
        model: _,
    } = request;
    let image = image::load_from_memory(&image_data)?;
    let vector = embedder.embed_image(&image)?;
    Ok(Embedding {
        id: uuid::Uuid::new_v4().to_string(),
        vector,
        model: ModelType::EfficientNet,
        backend: BackendType::Burn,
    })
}

#[derive(Debug)]
pub struct EmbeddingRequest {
    pub image_data: Vec<u8>,
    pub backend: BackendType,
    pub model: ModelType,
}
#[derive(Debug)]

pub struct Embedding {
    pub id: String,
    pub vector: Vec<f32>,
    pub model: ModelType,
    pub backend: BackendType,
}
#[derive(Debug)]
pub enum ModelType {
    EfficientNet,
}
#[derive(Debug)]
pub enum BackendType {
    Ort,
    Burn,
}
