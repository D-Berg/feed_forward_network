use std::default;
use std::fmt::format;
// use std::io::Write;
use std::{io::Read};



// used to print colorized std out 
use colored::Colorize;
use tch::nn::init::NonLinearity;



// progressbar 
use std::time::Duration;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};


//pytorch
use tch::{Tensor, Device, Cuda, Kind::Double, Kind::Float};
use tch::nn::{Module, self, OptimizerConfig};


fn net(vs: &nn::Path) -> impl Module {

    let in_dim= [28, 28];
    nn::seq()
        .add(nn::linear(vs, 784, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 128, 10, Default::default()))
      
}



/// A Feed Forward Network to be used on mnist data
/// Input a 28*28 = 784.
/// output a 10 length vector of probs 
/// have a function that take the index of max element and return that to 
/// get classification.





/// Function to be used by get data functions 
/// to minimize code repetition.
fn read_in_data(file_name: &str) -> Vec<u8> {

    let mut file = std::fs::File::open(file_name)
        .expect("Error opening file");

    let mut buffer: Vec<u8> = Vec::new();

    file.read_to_end(&mut buffer).expect("Error reading in file");

    return buffer;
}



fn get_training_labels() -> Tensor {
    let file_name: &str = "mnist_data/train_set_labels";

    return get_labels(file_name);

}

fn get_test_labels() -> Tensor {
    let file_name: &str = "mnist_data/test_set_labels";

    return get_labels(file_name);

}

/// Extracts data useful data from read_in_data 
/// and returns training labels as a vector.
fn get_labels(file_name: &str) -> Tensor {

    // we get unclean data from reading the file
    // we need to extract the useful information 

    let un_clean_data: Vec<u8> = read_in_data(file_name);

    //length of unclean data
    let n: usize = un_clean_data.len();

    let clean_data: Vec<u8>  = un_clean_data[8..n].try_into().unwrap();

    
    let mut clean_data_i64: Vec<i64> = vec![0; clean_data.len()];

    // convert to f64
    for i in 0..clean_data.len() {
        clean_data_i64[i] = clean_data[i] as i64;
    }
    let data_array: Tensor = Tensor::from_slice(&clean_data_i64);

    return data_array;

}


fn get_training_images() -> Tensor {

    let file_name: &str = "mnist_data/train_set_images";

    return get_images(file_name); 

}

fn get_test_images() -> Tensor {
    let file_name: &str = "mnist_data/test_set_images";

    return get_images(file_name); 
}



fn get_images(file_name: &str) -> Tensor {

    let un_clean_data: Vec<u8> = read_in_data(file_name);


    // dbg!(&un_clean_data[0..4]);

    let n: usize = un_clean_data.len();

    let n_images_binary: &[u8; 4] = &un_clean_data[4..8].try_into().unwrap();
    
    let n_images: usize = u32::from_be_bytes(*n_images_binary) as usize;
    // dbg!(n_images);

    let n_rows: usize = un_clean_data[11] as usize;
    // dbg!(n_rows);

    let n_cols: usize = un_clean_data[15] as usize;
    // dbg!(n_cols);


    let clean_data: Vec<u8> = un_clean_data[16..n].try_into().unwrap();

    // dbg!(clean_data.len() as f64 /  784.0);


    let mut clean_data_f64: Vec<f64> = vec![0.0; clean_data.len()];

    // convert to f64
    for i in 0..clean_data.len() {
        clean_data_f64[i] = clean_data[i] as f64;
    }

    // convert to array

    let images: Tensor = Tensor::from_slice(&clean_data_f64).reshape([n_images as i64, n_rows as i64, n_cols as i64]);

    let images = images.to_dtype(Float, true, true);

    return images;

}

/// Prints one picture
fn print_picture(
    image_index: usize, 
    data: &Tensor,
    labels: &Tensor,
) {

    let image: Tensor = data.select(0, image_index as i64);

    let label: u8 = labels.double_value(&[image_index as i64]) as u8;

    
    let image_shape: &[i64] = &image.size();

    let n_rows: usize = image_shape[0] as usize;
    let n_cols: usize = image_shape[1] as usize;

    println!("Picture of number {}", label);

    // https://notes.burke.libbey.me/ansi-escape-codes/


    // let stdout = std::io::stdout();
    // let mut handle = std::io::BufWriter::new(stdout);
    
    for i in 0..n_rows {
        for j in 0..n_cols {

            

            let pix_val: u8 = image.double_value(&[i as i64, j as i64]) as u8;

            // print!("{}", pixel.truecolor(pix_val, pix_val, pix_val));

            print!("\x1b[48;2;{};{};{}m  ", pix_val, pix_val, pix_val);
        }

        print!("\x1b[0m\n");
    }

    println!("\n");

    
}


fn main() {

    let m = tch::vision::mnist::load_dir("data").unwrap();

    dbg!(&m.train_images);
    dbg!(&m.train_labels);

    let device: Device = if Cuda::is_available() {
        Device::Cuda(0)
    
    } else {
        Device::Cpu
    };


    let training_labels: Tensor = get_training_labels().to_device(device);
    let training_images: Tensor = get_training_images().to_device(device);

    dbg!(&training_images);
    dbg!(&training_labels);

    let test_images: Tensor = get_test_images().to_device(device);
    let test_labels: Tensor = get_test_labels().to_device(device);

    dbg!(&test_images);
    dbg!(&test_labels);

    let scaled_training_images: Tensor = &training_images / 255.0;
    let scaled_test_images: Tensor = &test_images / 255.0;
    
    for i in 0..5 {
        print_picture(i, &training_images, &training_labels);
    }


    let flattened_train_images = scaled_training_images.reshape([60000, 784]);

    let flattened_test_images = scaled_test_images.reshape([10000, 784]);
    
    
    let vs = nn::VarStore::new(device);

    //neural net
    let net = net(&vs.root());

    //optimizer
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();


    let n_epochs: usize = 200;

    let pb = ProgressBar::new(n_epochs as u64);

    let sty = ProgressStyle::with_template(
        "Eta: {eta} {bar:50.yellow} {pos:>7}/{len:5} {elapsed:6} {msg:10.yellow}",
    )
    .unwrap();

    pb.set_style(sty);

    

    for epoch in 0..n_epochs {
        let loss = net.forward(&flattened_train_images).cross_entropy_for_logits(&training_labels);

        opt.backward_step(&loss);

        let test_accuracy = net.forward(&flattened_test_images).accuracy_for_logits(&test_labels);

        pb.set_message(
            format!(
                "epoch: {:4}, Train loss: {:8.5} test acc: {:5.2}%",
                epoch + 1,
                f64::try_from(&loss).expect("Failed to conv to f64"),
                100.0 * f64::try_from(&test_accuracy).expect("Failed to conv to f64")
            )
        );

        pb.inc(1);
    }

    pb.finish();

} 



// #[cfg(test)]
// mod tests {
//     use ndarray::{NewAxis, Array2, Array1};

//     use crate::{relu, FeedForwardNetwork};
//     use crate::Layer::*;
//     use crate::*;


//     #[test]
//     fn relu_activation_works() {
//         use relu;

//         let test1: f64 = 5.0;
//         let test2: f64 = -2.0;
//         let result1: f64 = relu(test1);
//         let result2: f64 = relu(test2);
//         assert_eq!(result1, test1);
//         assert_eq!(result2, 0.0);
//     }


//     // 1. create neural net with add_layer
//     // 2. call initialize weights and biases 
//     // 3. assert shapes of w and b are correct

//     #[test]
//     fn test_initialize_weights_and_biases() {
//         let n_0: usize = 2;
//         let n_1: usize = 4; // 
//         let n_2: usize = 5; //


//         let mut test_network: FeedForwardNetwork = FeedForwardNetwork::new()
//             .add_layer(ReLU(n_1))
//             .add_layer(SoftMax(n_2));

//         test_network.initialize_weights_and_biases(n_0);

//         let weights:Vec<Array2<f64>> = test_network.weights;
//         let biases: Vec<Array2<f64>> = test_network.biases;


//         let w_1_shape: &[usize] = weights[0].shape();
//         let w_2_shape: &[usize] = weights[1].shape();
//         let b_1_shape: &[usize] = biases[0].shape();
//         let b_2_shape: &[usize] = biases[1].shape();

//         assert_eq!(w_1_shape, &[n_1, n_0]);
//         assert_eq!(w_2_shape, &[n_2, n_1]);
//         assert_eq!(b_1_shape, &[n_1, 1]);
//         assert_eq!(b_2_shape, &[n_2, 1]);

        
//     }

//     #[test]
//     fn test_softmax() {

//         let e_1 = 0.2;
//         let e_2 = 0.4;


//         let vec: Array2<f64> = Array2::from_shape_vec((2, 1), vec![e_1, e_2]).expect("failed to create test array");


//         let soft_m_func_result = softmax(&vec);


//         let answer_1 = soft_m_func_result[[0, 0]];
//         let answer_2 = soft_m_func_result[[1, 0]];

//         let correct_1 = e_1.exp() / (e_1.exp() + e_2.exp());

//         let correct_2 = e_2.exp() / (e_1.exp() + e_2.exp());

        

//         // assert_eq!(correct_1 + correct_2, 1.0);
//         assert_eq!(answer_1 + answer_2, 1.0);
//         assert_eq!(answer_1, correct_1);
//         assert_eq!(answer_2, correct_2);

        

//     }


// }