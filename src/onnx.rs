#[cfg(feature = "ort")]
pub struct OrtEmbedder {
    model: Session,
    convert: ImageConvert,
}

//pub use image::DynamicImage;
#[cfg(feature = "ort")]
pub use ort::session::Session;
#[cfg(feature = "ort")]
pub use ort::Error;

#[cfg(feature = "ort")]
pub fn efficientnet_apple_session() -> EmbedderResult<Session> {
    use ort::execution_providers::{CoreMLExecutionProvider, XNNPACKExecutionProvider};
    use ort::session::builder::GraphOptimizationLevel;
    tracing_subscriber::fmt::init();

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers([
            CoreMLExecutionProvider::default().build(),
            XNNPACKExecutionProvider::default().build(),
        ])?
        .commit_from_memory(EFFICIENTNET)?;
    Ok(session)
}

use crate::DynamicImage;
use crate::EmbedderResult;
use crate::ImageConvert;
use crate::EFFICIENTNET;
use crate::IMAGENET_DEFAULT_CONFIG;
#[cfg(feature = "ort")]
impl OrtEmbedder {
    pub fn new(session: Session) -> Self {
        Self {
            model: session,
            convert: IMAGENET_DEFAULT_CONFIG,
        }
    }
    pub fn embed_image(&self, image: &DynamicImage) -> EmbedderResult<Vec<f32>> {
        let input = self.convert.ort_value(image)?;
        let prediction = self.model.run(ort::inputs!["actual_input" => input]?)?;
        let output = prediction["output"].try_extract_tensor::<f32>()?;

        let embedding: Vec<f32> = output.into_iter().cloned().collect();
        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "burn")]
    use crate::{cosine, EfficientNetEmbedder};

    use super::*;
    use crate::cosine::cosine_similarity;
    #[cfg(feature = "burn")]
    use burn_ndarray::NdArray;

    #[cfg(feature = "burn")]
    #[test]
    fn result_consistency() {
        let image = image::open("test_resources/nutella2.jpg").unwrap();
        let session = Session::builder()
            .unwrap()
            // .with_optimization_level(GraphOptimizationLevel::Level3)
            //  .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_file("src/model/efficientnet.onnx")
            .unwrap();
        let ort_embedder = OrtEmbedder::new(session);
        let burn_embedder = EfficientNetEmbedder::<NdArray>::new().unwrap();

        let ort_embedding = ort_embedder.embed_image(&image).unwrap();
        let burn_embedding = burn_embedder.embed_image(&image).unwrap();
        let similarity = cosine_similarity(&ort_embedding, &burn_embedding).unwrap();

        let wanted_similarity = "1.0";
        println!("GOT: {similarity} WANT: {wanted_similarity}");

        assert!(format!("{similarity}").contains(wanted_similarity));
    }

    ///This tests if the same/similar object has a higher similarity to the same class despite having a different background
    fn off_mark(coke1: Vec<f32>, coke2: Vec<f32>, nutella1: Vec<f32>, nutella2: Vec<f32>) {
        let c1c2 = cosine_similarity(&coke1, &coke2).unwrap();
        let n1n2 = cosine_similarity(&nutella1, &nutella2).unwrap();

        let c1n1 = cosine_similarity(&coke1, &nutella1).unwrap();
        let c1n2 = cosine_similarity(&coke1, &nutella2).unwrap();
        let c2n1 = cosine_similarity(&coke2, &nutella1).unwrap();
        let c2n2 = cosine_similarity(&coke2, &nutella2).unwrap();

        println!(" C1C2 {c1c2} \n N1N2 {n1n2} \n C1N1 {c1n1} \n C1N2 {c1n2} \n C2N1 {c2n1} \n C2N2 {c2n2} ");
        assert!(
            c1c2 > c1n1,
            "Coke1 should be more similar to Coke2 than to c1n1"
        );
        assert!(
            c1c2 > c1n2,
            "Coke1 should be more similar to Coke2 than to any c1n2"
        );
        assert!(
            c1c2 > c2n1,
            "Coke1 should be more similar to Coke2 than to c2n1"
        );
        assert!(
            c1c2 > c2n2,
            "Coke1 should be more similar to Coke2 than to c2n2"
        );

        assert!(
            n1n2 > c1n1,
            "Nutella1 should be more similar to Nutella2 than to c1n1"
        );
        assert!(
            n1n2 > c1n2,
            "Nutella1 should be more similar to Nutella2 than to c1n2"
        );
        assert!(
            n1n2 > c2n1,
            "Nutella1 should be more similar to Nutella2 than to c2n1"
        );
        assert!(
            n1n2 > c2n2,
            "Nutella1 should be more similar to Nutella2 than to c2n1"
        );
    }
    #[cfg(feature = "ort")]
    #[test]
    fn test_efficientnet_ort() {
        use ort::session::builder::GraphOptimizationLevel;
        let coke1 = image::open("test_resources/coke1.jpg").unwrap();
        let coke2 = image::open("test_resources/coke2.jpg").unwrap();
        let nutella1 = image::open("test_resources/nutella1.jpg").unwrap();
        let nutella2 = image::open("test_resources/nutella2.jpg").unwrap();
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_file("src/model/efficientnet.onnx")
            .unwrap();
        let embedder = OrtEmbedder::new(session);

        let coke1 = embedder.embed_image(&coke1).unwrap();
        let coke2 = embedder.embed_image(&coke2).unwrap();
        let nutella1 = embedder.embed_image(&nutella1).unwrap();
        let nutella2 = embedder.embed_image(&nutella2).unwrap();

        off_mark(coke1, coke2, nutella1, nutella2);
    }
}
