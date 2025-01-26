use image::DynamicImage;
pub trait Embedder<T> {
    fn embed_image(image: &DynamicImage) -> Vec<f32>;
}
