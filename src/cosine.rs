/// Computes the cosine similarity between two vectors of `f32` values.
///
/// # Arguments
///
/// * `vec1` - A vector of `f32` values.
/// * `vec2` - Another vector of `f32` values.
///
/// # Returns
///
/// A `Result<f32, &'static str>` where:
/// - `Ok(f32)` contains the cosine similarity if the computation succeeds.
/// - `Err(&str)` contains an error message if the vectors are of different lengths or one of the vectors is zero.
#[allow(dead_code)]
pub fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> Result<f32, &'static str> {
    if vec1.len() != vec2.len() {
        return Err("Vectors must be of the same length");
    }

    let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let magnitude1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude1 == 0.0 || magnitude2 == 0.0 {
        return Err("One of the vectors has zero magnitude");
    }

    Ok(dot_product / (magnitude1 * magnitude2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let similarity = cosine_similarity(&vec1, &vec2).unwrap();
        assert!((similarity - 0.9746318).abs() < 1e-6);
    }

    #[test]
    fn test_zero_magnitude() {
        let vec1 = vec![0.0, 0.0, 0.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        assert!(cosine_similarity(&vec1, &vec2).is_err());
    }

    #[test]
    fn test_different_lengths() {
        let vec1 = vec![1.0, 2.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        assert!(cosine_similarity(&vec1, &vec2).is_err());
    }
}
