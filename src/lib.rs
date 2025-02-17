uniffi::include_scaffolding!("embedder");

#[cfg(feature = "burn")]
use burn::{
    prelude::Backend,
    tensor::{Device, Tensor},
};
#[cfg(feature = "burn")]
#[allow(unused_imports)]
pub use burn_ndarray::{NdArray, NdArrayDevice};
#[allow(unused_imports)]
#[cfg(feature = "burn")]
pub use burn_wgpu::{Wgpu, WgpuDevice};
#[cfg(feature = "burn")]
use error::EmbedderError;
use error::EmbedderResult;
use image::imageops::FilterType;
pub use image::DynamicImage;
//use mobile::{BackendType, Embedding, EmbeddingRequest, ModelType};
#[cfg(feature = "burn")]
use model::efficientnet::Model;
#[cfg(feature = "ort")]
use ort::value::{TensorValueType, Value};

mod cosine;
pub mod embedder;
pub mod error;
pub mod mobile;
#[allow(unused_imports)]
pub use mobile::*;
#[cfg(feature = "burn")]
mod model;
pub mod onnx;

#[cfg(feature = "ort")]
pub const EFFICIENTNET: &[u8] = include_bytes!("model/efficientnet.onnx");

const IMAGENET_DEFAULT_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_DEFAULT_STD: [f32; 3] = [0.229, 0.224, 0.225];
const IMAGENET_DEFAULT_CONFIG: ImageConvert = ImageConvert {
    channels: 3,
    width: 256,
    height: 256,
    crop: 224,
    mean: IMAGENET_DEFAULT_MEAN,
    std: IMAGENET_DEFAULT_STD,
    interpolation: FilterType::CatmullRom,
};

pub enum Parallelize {
    None,
    PerImage,
    PerPixel,
}

//https://arxiv.org/pdf/1712.07629v4

pub struct ImageConvert {
    //pub batches: u16,
    pub channels: u8,
    pub width: u32,
    pub height: u32,
    pub crop: u16, //central crop of the image
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub interpolation: image::imageops::FilterType,
}

impl ImageConvert {
    #[cfg(feature = "ort")]
    pub fn ort_value(&self, image: &DynamicImage) -> EmbedderResult<Value<TensorValueType<f32>>> {
        let normalized_data = self.create_data(image);
        let tensor_shape = vec![
            1,
            self.channels as usize,
            self.crop as usize,
            self.crop as usize,
        ];
        let tensor_args = (tensor_shape, normalized_data);
        let input_array = ort::value::Tensor::from_array(tensor_args)?;
        Ok(input_array)
    }
    fn create_data(&self, image: &DynamicImage) -> Vec<f32> {
        let resized = image.resize_exact(self.width, self.height, self.interpolation);

        // Central crop to 224 x 224
        let (width, height) = (resized.width(), resized.height());
        let crop_x = (width - 224) / 2;
        let crop_y = (height - 224) / 2;
        let cropped = resized.crop_imm(crop_x, crop_y, 224, 224);

        // Convert to f32 array and rescale to [0.0, 1.0]
        let cropped_rgb = cropped.to_rgb8();
        let (width, height) = (cropped.width(), cropped.height());

        let mut data = Vec::new();

        for pixel in cropped_rgb.pixels() {
            data.push(pixel[0] as f32 / 255.0); // Red channel
            data.push(pixel[1] as f32 / 255.0); // Green channel
            data.push(pixel[2] as f32 / 255.0); // Blue channel
        }

        // Reshape into [3, 224, 224] and normalize
        let mean = self.mean;
        let std = self.std;

        let mut normalized_data = vec![0.0; data.len()];
        for c in 0..3 {
            for i in 0..(width * height) as usize {
                normalized_data[c * (width * height) as usize + i] =
                    (data[i * 3 + c] - mean[c]) / std[c];
            }
        }
        normalized_data
    }
    #[cfg(feature = "burn")]
    pub fn burn_tensor<B: Backend>(
        &self,
        image: &DynamicImage,
        device: &Device<B>,
    ) -> Tensor<B, 4> {
        let normalized_data = self.create_data(image);
        Tensor::<B, 1>::from_floats(normalized_data.as_slice(), &device).reshape([
            1,
            self.channels as usize,
            self.crop as usize,
            self.crop as usize,
        ])
    }
}
#[cfg(feature = "burn")]
unsafe impl<B: Backend> Send for EfficientNetEmbedder<B> {}
#[cfg(feature = "burn")]
unsafe impl<B: Backend> Sync for EfficientNetEmbedder<B> {}
#[cfg(feature = "burn")]
pub struct EfficientNetEmbedder<B: Backend> {
    model: Model<B>,
    device: Device<B>,
    convert: ImageConvert,
}
#[cfg(feature = "burn")]
impl<B: Backend> EfficientNetEmbedder<B> {
    pub fn new() -> EmbedderResult<Self> {
        /*
        The inference transforms are available at EfficientNet_B0_Weights.IMAGENET1K_V1.transforms and perform the following preprocessing operations:
        Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects.
        The images are resized to resize_size=[256] using interpolation=InterpolationMode.BICUBIC, followed by a central crop of crop_size=[224].
        Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
        */
        let device = B::Device::default();
        let model = Model::<B>::default();
        println!("{:?}", model);
        let convert = IMAGENET_DEFAULT_CONFIG;
        Ok(EfficientNetEmbedder {
            model,
            device,
            convert,
        })
    }
    pub fn embed_image(&self, image: &DynamicImage) -> EmbedderResult<Vec<f32>> {
        let tensor = self.convert.burn_tensor::<B>(image, &self.device);
        let result = self
            .model
            .forward(tensor)
            .into_data()
            .convert::<f32>()
            .to_vec();
        let embedding: Vec<f32> = result.map_err(|e| EmbedderError::Burn(e))?;

        Ok(embedding)
    }
    pub async fn embed_image_async(&self, image: &DynamicImage) -> EmbedderResult<Vec<f32>> {
        let tensor = self.convert.burn_tensor::<B>(image, &self.device);
        let result = self
            .model
            .forward(tensor)
            .into_data_async()
            .await
            .convert::<f32>()
            .to_vec();
        let embedding: Vec<f32> = result.map_err(|e| EmbedderError::Burn(e))?;

        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cosine::cosine_similarity;

    ///This tests if the same/similar object has a higher similarity to the same class despite having a different background
    fn _off_mark(coke1: Vec<f32>, coke2: Vec<f32>, nutella1: Vec<f32>, nutella2: Vec<f32>) {
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

    ///Tests if the EfficientNet Encoder passes offMark
    #[cfg(feature = "burn")]
    #[test]
    fn test_efficientnet() {
        /*
        Nutella1 <-> Nutella2: Ok(0.3972155)
        Coke1 <-> Coke2: Ok(0.6631188)
        Coke1 <-> Nutella2: Ok(0.2386467)
        Coke1 <-> Nutella1: Ok(0.23746115)
        Coke2 <-> Nutella2: Ok(0.29427624)
        Coke2 <-> Nutella1: Ok(0.2570323)
        */
        let coke1 = image::open("test_resources/coke1.jpg").unwrap();
        let coke2 = image::open("test_resources/coke2.jpg").unwrap();
        let nutella1 = image::open("test_resources/nutella1.jpg").unwrap();
        let nutella2 = image::open("test_resources/nutella2.jpg").unwrap();
        let embedder = EfficientNetEmbedder::<NdArray>::new().unwrap();

        let coke1 = embedder.embed_image(&coke1).unwrap();
        let coke2 = embedder.embed_image(&coke2).unwrap();
        let nutella1 = embedder.embed_image(&nutella1).unwrap();
        let nutella2 = embedder.embed_image(&nutella2).unwrap();
        _off_mark(coke1, coke2, nutella1, nutella2);
    }
}
