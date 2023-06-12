use std::io::Write;
use std::{io::Read};
use ndarray::*;
// use std::io::Write;
use Layer::*;
use LossFunction::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

// used to print colorized std out 
use colored::Colorize;
use ndarray_rand::rand_distr::num_traits::{pow, Pow};

// progressbar 
use std::time::Duration;


// threading 
use std::thread;
use std::thread::JoinHandle;
use std::sync::{Arc, Mutex, MutexGuard};

/// A Feed Forward Network to be used on mnist data
/// Input a 28*28 = 784.
/// output a 10 length vector of probs 
/// have a function that take the index of max element and return that to 
/// get classification.
#[derive(Debug)]
struct FeedForwardNetwork {
    layers: Vec<Layer>,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,

}


impl FeedForwardNetwork {

    // create a new ffn object
    fn new() -> FeedForwardNetwork {
        FeedForwardNetwork {
            layers: Vec::new(),
            weights: Vec::new(),
            biases: Vec::new(),
        }

    }


    /// Train the network on the training images and labels
    pub fn train(
        &mut self,
        epochs: usize,
        training_images: &Array3<f64>,
        training_labels: &Array1<u8>,
        loss_function: LossFunction  
    ) {

        println!("{}", "Start training...\n".red());

        let mut total_training_time: Duration= Duration::new(0, 0);

        //learning rate 
        let eta: f64 = 1.0;

        let shape_images: &[usize] = &training_images.shape();

        let n_images: usize = shape_images[0];

        let n_neurons_input_layer: usize = shape_images[1] * shape_images[2];

        self.initialize_weights_and_biases(n_neurons_input_layer);

        let mut costs: Vec<f64> = Vec::with_capacity(n_images);

        'epoch_loop: for epoch in 0..epochs {


            println!("{} {}/{}", "Epoch:".blue(), epoch + 1, epochs);
            // std::io::stdout().flush().expect("Failed to flush stdout");
            

            let mut number_correct_predictions: u32 = 0;

            // initialize gradients 
            let mut weight_gradients: Vec<Array2<f64>> = Vec::with_capacity(n_images);
            let mut bias_gradients: Vec<Array2<f64>> = Vec::with_capacity(n_images);

            for l in 0..self.weights.len() {

                let weight_shape: &[usize] = &self.weights[l].shape();
                let bias_shape = &self.biases[l].shape();

                weight_gradients.push(Array2::<f64>::zeros([weight_shape[0], weight_shape[1]]));

                bias_gradients.push(Array2::<f64>::zeros([bias_shape[0], bias_shape[1]]));

            }

            let pb: indicatif::ProgressBar = indicatif::ProgressBar::new(n_images as u64);

            // TODO: parallelize this
            for i in 0..n_images {

                let image: Array2<f64> = training_images.slice(s![i, .., ..]).to_owned();
                

                let label = training_labels[[i]];

                // calculate activations for image 
                // Vector of (a^0, ... a^L)
                // where a^l is a (n_l x 1) vector
                // We will have a^0 which is just the image input reshaped
                let activations: Vec<Array2<f64>> = self.feed_forward(image);

                let predictions_probs: &Array2<f64> = &activations[activations.len()-1];

                let predictions_len: usize = predictions_probs.shape()[0];

                let mut max_prob: f64 = 0.0;

                let mut predicted_label: u8 = 0;

                for j in 0..predictions_len {

                    let prediction_prob: f64 = predictions_probs[[j, 0]];

                    if prediction_prob >=  max_prob {
                        predicted_label = j as u8;
                        max_prob = prediction_prob;

                    }

                }

                if predicted_label == label {
                    number_correct_predictions += 1;
                }

                // dbg!(&activations[0]);

                // backpropegate image in network to get deltas

                let deltas: Vec<Array2<f64>> = self.backpropegate(
                    &activations, 
                    &label, 
                    &loss_function
                );

               
                // calculate gradient for dC_x/dW and dC_x/db (matrix form)
                let L = self.layers.len();
                
                for l in (0..L).rev() {

                    let a_l_min_1: &Array2<f64> = &activations[l];

                    let weight_grad_x: Array2<f64> = deltas[l].dot(&a_l_min_1.t());


                    weight_gradients[l] = &weight_gradients[l] + weight_grad_x;

                    bias_gradients[l] = deltas[l].to_owned();
                    

                }  

                // creates vec [0, 0, 0 , 1, ..., 0]
                let mut label_vec: Vec<f64> = vec![0.0; 10]; 

                label_vec[label as usize] = 1.0;

                let y: Array2<f64> = Array2::<f64>::from_shape_vec((label_vec.len(), 1), label_vec).expect("failed to create y");

                let a_L = &activations[activations.len()-1];

                // dbg!(label);

                // dbg!(a_L);

                let cost: f64 = match loss_function {
                    LeastSquares => {

                        (a_L - y).mapv(|a: f64| a.pow(2)).sum()
                        
                    },
                    CrossEntropy => -a_L[[label as usize, 0]].ln(),
                };

                // dbg!(cost);

                costs.push(cost);

                
                pb.inc(1);

            }

            

            // Update weights W^l = W^l - grad(W^l)
            // remember grad(W^l) = mean(dC_x/dW^l)

            self.sgd(&weight_gradients, &bias_gradients, n_images, &eta);


            
            // Update weights b^l = b^l - grad(b^l)
            // grad(b^l) = mean(dC_x/db^l) = mean(delta^l_x)

            let accuracy = number_correct_predictions as f64 / n_images as f64;
            
            let total_cost: f64 = costs.iter().sum();

            let average_cost: f64 = total_cost / n_images as f64;


            pb.finish();
            let epoch_time: f64 =  pb.elapsed().as_secs_f64();

            // let epoch_time: f64 = epoch_time.try_into().expect("Failed to convert");

            println!(
                "{} {:.2}s, {} {:.2}, {}: {:.2}", 
                "Time:".blue(),
                epoch_time,
                "Accuracy:".blue(),
                accuracy,
                "Loss".blue(),
                average_cost
            );

            total_training_time += pb.elapsed();
            
            print!("\n");
        }


        println!("\n{} {:?}", "Finished training in:".green(), total_training_time);

    }

    /// should just take one image  
    /// Calculate a^l = activation(W^l * a^{l-1} + b_l)    
    fn feed_forward(
        &mut self,
        image: Array2<f64>
    ) -> Vec<Array2<f64>> {

        // index of last layer
        let L: usize = self.layers.len();

        let mut activations: Vec<Array2<f64>> = Vec::with_capacity(L);

        // assume first layer is always flatten for now 

        let a_0: Array2<f64> = image.into_shape((784, 1)).unwrap();

        activations.push(a_0);

        // activations = [a_1, a_2, ..., a_L]
        // Weigths = [w_1, w_2, w_L]
        // TODO: Re-check indexing
        for l in 1..(L + 1) {

            let layer = self.layers[l - 1];

            let w_l: &Array2<f64> = &self.weights[l - 1];

            let b_l: &Array2<f64> = &self.biases[l - 1];

            let z_l: Array2<f64> = w_l.dot(&activations[l - 1]) + b_l;

            let a_l: Array2<f64> = match layer {
                ReLU(_) => z_l.mapv(relu),
                SoftMax(_) => softmax(&z_l),
            };

            activations.push(a_l);

        }

        return activations;

    }

    /// Calculate d^L = a^L * (1 - a^L)* (a^L - y)
    /// d^l = a^l * (1 - a^l) * (W^{l + 1} * d^{l + 1}
    fn backpropegate(
        &mut self,
        activations: &Vec<Array2<f64>>,
        label: &u8,
        loss_function: &LossFunction

    ) -> Vec<Array2<f64>> {

        let L: usize = self.layers.len() - 1;


        let mut deltas: Vec<Array2<f64>> = vec![Array2::<f64>::zeros((0, 0)); L + 1];


        let weights: &Vec<Array2<f64>> = &self.weights;
        let biases: &Vec<Array2<f64>> = &self.biases;

        let label_f64: f64 = label.clone() as f64;

        let n_last_layer: usize = match self.layers[L] {
            ReLU(n) => n,
            SoftMax(n) => n,
        };

        // creates vec [0, 0, 0 , 1, ..., 0]
        let mut label_vec: Vec<f64> = vec![0.0; n_last_layer]; 

        label_vec[*label as usize] = 1.0;

        let y: Array2<f64> = Array2::<f64>::from_shape_vec((label_vec.len(), 1), label_vec).expect("failed to create y");

        //for Layer L

        let a_L: &Array2<f64> = &activations[L + 1];
        let w_L: &Array2<f64> = &weights[L];
        let b_L: &Array2<f64> = &biases[L];

        let z_L: Array2<f64> = w_L.dot(&activations[L]) + b_L;


        let sigma_prime_L: Array2<f64> = match self.layers[L] {
            ReLU(_) => z_L.mapv(relu_prime),
            SoftMax(_) => softmax_prime(&z_L),            
        };


        let delta_L: Array2<f64> = match loss_function {
            LeastSquares => 2.0 * (a_L - y) * sigma_prime_L,
            CrossEntropy => {
               a_L - y
            },
             
        };

        deltas[L] = delta_L;

        // TODO: Reverese this 
        for i in 0..L {
            let l: usize = (L - 1) - i;
           

            let w_l: &Array2<f64>  = &weights[l];
            let b_l: &Array2<f64> = &biases[l];

            let w_l_plus_1: &Array2<f64> = &weights[l + 1];

            let a_l_munus_1: &Array2<f64> = &activations[l];

            //index of activations i + 1
            let z_l: Array2<f64> = w_l.dot(a_l_munus_1) + b_l;


            let sigma_prime_l: Array2<f64> = match self.layers[l] {
                ReLU(_) => z_l.mapv(relu_prime),
                SoftMax(_) => softmax_prime(&z_l),
            };


            let delta_l: Array2<f64> = w_l_plus_1.t().dot(&deltas[l + 1]) * sigma_prime_l;
            

            deltas[l] = delta_l;

        }

        return deltas;
    }


    fn sgd(
        &mut self,
        weight_gradients: &Vec<Array2<f64>>,
        bias_gradients: &Vec<Array2<f64>>,
        n_images: usize,
        eta: &f64
    ) {

        let L = self.layers.len();

        for l in (0..L).rev() {

            let w_l: &Array2<f64> = &self.weights[l];
            let b_l: &Array2<f64> = &self.biases[l];

            self.weights[l] = w_l - eta / (n_images as f64) * &weight_gradients[l];
            self.biases[l] = b_l - eta / (n_images as f64) * &bias_gradients[l];

        }
    }

    // returns predictions 
    pub fn test(
        &mut self,
        test_images: &Array3<f64>,
    ) -> Array1<u8> {

        println!("\n{}", "Starting test...".red());
        
        let shape_images: &[usize] = &test_images.shape();

        let n_images: usize = shape_images[0];

        let n_neurons_input_layer: usize = shape_images[1] * shape_images[2];

        
        let pb = indicatif::ProgressBar::new(n_images as u64);

        let mut predicted_labels: Array1<u8> = Array1::<u8>::zeros(n_images);
        
        for i in 0..n_images {

            let image: Array2<f64> = test_images.slice(s![i, .., ..]).to_owned();

            
    
            let activations: Vec<Array2<f64>> = self.feed_forward(image);

            // get predictions (last activation layer)
            let predictions_probs: &Array2<f64> = &activations[activations.len() - 1];

            
            let predictions_len: usize = predictions_probs.shape()[0];

            let mut max_prob: f64 = 0.0;

            let mut predicted_label: u8 = 0;

            for j in 0..predictions_len {

                let prediction_prob: f64 = predictions_probs[[j, 0]];

                if prediction_prob >=  max_prob {
                    predicted_label = j as u8;
                    max_prob = prediction_prob;

                }

            }

            predicted_labels[[i]] = predicted_label;


            pb.inc(1);
        }

        pb.finish();
        println!("{}", "Finished testing".green());

        return predicted_labels;


    }

    /// adds a layer to to the network
    pub fn add_layer(mut self, layer: Layer) -> Self {

        self.layers.push(layer);

        return self;

    }

    fn initialize_weights_and_biases(&mut self, n_0: usize) -> &Self {

        // TODO: initiate weights and biases to 0 (for now)

        let mut  n_previous_layer: usize = n_0;
        
        for layer in self.layers.iter() {

            // n_l
            let n_current_layer: usize = match layer { 
                ReLU(n) => *n,
                SoftMax(n) => *n,
            };

            let w_l_shape:[usize; 2] = [n_current_layer, n_previous_layer];

            let w_l: Array2<f64> = Array2::<f64>::random(
                w_l_shape,
                 Normal::new(0.0, 0.1).unwrap()
            );


            let b_l: Array2<f64> = Array2::<f64>::random(
                (n_current_layer, 1),
                 Normal::new(0.0, 0.1).unwrap()
            );


            self.weights.push(w_l);
            self.biases.push(b_l);

            // update the n_{l-1}
            n_previous_layer = n_current_layer;


        }

        // dbg!(&self.weights[1]);

        return self;

    }

    fn get_accuracy(
        &self,
        test_labels: &Array1<u8>,
        predicted_labels: &Array1<u8>
    ) -> f64 {

        let n_labels = test_labels.shape()[0];

        let mut n_correct: u32 = 0;

        for i in 0..n_labels {

            if predicted_labels[[i]] == test_labels[[i]] {
                n_correct += 1;
            }

        }

        let accuracy = n_correct as f64 / n_labels as f64;


        return accuracy;

    }



}

#[derive(Debug, Clone, Copy)]
enum Layer{
    ReLU(usize),
    SoftMax(usize), 
} 

impl Layer {
    fn get_n_neurons(&self) -> usize {

        match self {
            ReLU(val) => *val,
            SoftMax(val) => *val,
        }
        
    }
}

fn relu(z: f64) -> f64 {
    f64::max(z, 0.0)
}

fn relu_prime(z: f64) -> f64 {
    if z == 0.0 {
        0.0
    } else {
        1.0
    } 
}


// take z_vec and return sig(z_vec)_i = e^(z_i) / sum(e^(z_vec))
fn softmax(z_vec: &Array2<f64>) -> Array2<f64> { 

    z_vec.mapv(f64::exp) / z_vec.mapv(f64::exp).sum()
    
}


fn softmax_prime(z_vec: &Array2<f64>) -> Array2<f64> {

    softmax(z_vec) * (1.0 - softmax(z_vec))

}


enum LossFunction {
    LeastSquares,
    CrossEntropy
}



/// Function to be used by get data functions 
/// to minimize code repetition.
fn read_in_data(file_name: &str) -> Vec<u8> {

    let mut file = std::fs::File::open(file_name)
        .expect("Error opening file");

    let mut buffer: Vec<u8> = Vec::new();

    file.read_to_end(&mut buffer).expect("Error reading in file");

    return buffer;
}



fn get_training_labels() -> Array1<u8> {
    let file_name: &str = "mnist_data/train_set_labels";

    return get_labels(file_name);

}

fn get_test_labels() -> Array1<u8> {
    let file_name: &str = "mnist_data/test_set_labels";

    return get_labels(file_name);

}

/// Extracts data useful data from read_in_data 
/// and returns training labels as a vector.
fn get_labels(file_name: &str) -> Array1<u8> {

    // we get unclean data from reading the file
    // we need to extract the useful information 

    let un_clean_data: Vec<u8> = read_in_data(file_name);

    //length of unclean data
    let n: usize = un_clean_data.len();

    let clean_data: Vec<u8>  = un_clean_data[8..n].try_into().unwrap();

    let data_array: Array1<u8> = Array::from(clean_data);

    return data_array;

}


fn get_training_images() -> Array3<f64> {

    let file_name: &str = "mnist_data/train_set_images";

    return get_images(file_name); 

}

fn get_test_images() -> Array3<f64> {
    let file_name: &str = "mnist_data/test_set_images";

    return get_images(file_name); 
}



fn get_images(file_name: &str) ->Array3<f64> {

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

    let data_array: Array3<f64> = Array::from_shape_vec(
        (n_images, n_rows, n_cols),
        clean_data_f64)
        .expect("Error: wrong shape");


    return data_array;

}

/// Prints one picture
fn print_picture(
    image_index: usize, 
    data: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
    labels: &ArrayBase<OwnedRepr<u8>, Dim<[usize; 1]>>,
) {

    let image_array: ArrayBase<ViewRepr<&f64>, Dim<[usize; 2]>> = data.slice(s![image_index, .., ..]);

    let label: u8 = labels[[image_index]];

    
    let image_shape: &[usize] = image_array.shape();

    let n_rows: usize = image_shape[0];
    let n_cols: usize = image_shape[1];

    println!("Picture of number {}", label);

    // https://notes.burke.libbey.me/ansi-escape-codes/


    // let stdout = std::io::stdout();
    // let mut handle = std::io::BufWriter::new(stdout);
    
    for i in 0..n_rows {
        for j in 0..n_cols {


            let pix_val: u8 = image_array[[i, j]] as u8;

            // print!("{}", pixel.truecolor(pix_val, pix_val, pix_val));

            print!("\x1b[48;2;{};{};{}m  ", pix_val, pix_val, pix_val);
        }

        print!("\x1b[0m\n");
    }

    println!("\n");

    
}


fn main() {

    let training_labels: Array1<u8> = get_training_labels();
    let training_images: Array3<f64> = get_training_images();

    let test_images: Array3<f64> = get_test_images();
    let test_labels: Array1<u8> = get_test_labels();

    let scaled_training_images: Array3<f64> = &training_images / 255.0;
    let scaled_test_images: Array3<f64> = &test_images / 255.0;
    
    // for i in 0..1 {
    //     print_picture(i, &training_images, &training_labels);
    // }

    // for i in 0..1 {
    //     print_picture(i, &test_images, &test_labels);
    // }

    let mut network: FeedForwardNetwork = FeedForwardNetwork::new()
        .add_layer(ReLU(128))
        .add_layer(SoftMax(10));


    
    network.train(
        10, 
        &scaled_training_images, 
        &training_labels,
        CrossEntropy
    );


    let predicted_labels: Array1<u8> = network.test(&scaled_test_images);
    

    for i in 0..5 {

        let predicted_label: u8 = predicted_labels[[i]];
        let correct_label: u8 = test_labels[[i]];

        if predicted_label == correct_label { // correct
            println!("{}, Predicted a {}", "Correct prediction".green(), predicted_label);
        } else {
            println!("{}, predicted a {}", "Wrong prediction".red(), predicted_label);
        }

        print_picture(i, &test_images, &test_labels);

        std::thread::sleep(Duration::from_secs(1));

        

    }

    let accuracy = network.get_accuracy(&test_labels, &predicted_labels);
    
    println!("Accuracy: {}", accuracy);


    
    // let test_image: Array2<f64> = scaled_test_images.slice(s![0, .., ..]).to_owned();



    // let prediction: &Array2<f64> = &network.feed_forward(test_image)[2];

    // println!("image of an {}", test_labels[[0]]);

    // println!("{:?}",prediction);

   
    

} 



#[cfg(test)]
mod tests {
    use ndarray::{NewAxis, Array2, Array1};

    use crate::{relu, FeedForwardNetwork};
    use crate::Layer::*;
    use crate::*;


    #[test]
    fn relu_activation_works() {
        use relu;

        let test1: f64 = 5.0;
        let test2: f64 = -2.0;
        let result1: f64 = relu(test1);
        let result2: f64 = relu(test2);
        assert_eq!(result1, test1);
        assert_eq!(result2, 0.0);
    }


    //TODO: Write a test for initialization of wieghts and biases.
    // 1. create neural net with add_layer
    // 2. call initialize weights and biases 
    // 3. assert shapes of w and b are correct

    #[test]
    fn test_initialize_weights_and_biases() {
        let n_0: usize = 2;
        let n_1: usize = 4; // 
        let n_2: usize = 5; //


        let mut test_network: FeedForwardNetwork = FeedForwardNetwork::new()
            .add_layer(ReLU(n_1))
            .add_layer(SoftMax(n_2));

        test_network.initialize_weights_and_biases(n_0);

        let weights:Vec<Array2<f64>> = test_network.weights;
        let biases: Vec<Array2<f64>> = test_network.biases;


        let w_1_shape: &[usize] = weights[0].shape();
        let w_2_shape: &[usize] = weights[1].shape();
        let b_1_shape: &[usize] = biases[0].shape();
        let b_2_shape: &[usize] = biases[1].shape();

        assert_eq!(w_1_shape, &[n_1, n_0]);
        assert_eq!(w_2_shape, &[n_2, n_1]);
        assert_eq!(b_1_shape, &[n_1, 1]);
        assert_eq!(b_2_shape, &[n_2, 1]);

        
    }

    #[test]
    fn test_softmax() {

        let e_1 = 0.2;
        let e_2 = 0.4;


        let vec: Array2<f64> = Array2::from_shape_vec((2, 1), vec![e_1, e_2]).expect("failed to create test array");


        let soft_m_func_result = softmax(&vec);


        let answer_1 = soft_m_func_result[[0, 0]];
        let answer_2 = soft_m_func_result[[1, 0]];

        let correct_1 = e_1.exp() / (e_1.exp() + e_2.exp());

        let correct_2 = e_2.exp() / (e_1.exp() + e_2.exp());

        

        // assert_eq!(correct_1 + correct_2, 1.0);
        assert_eq!(answer_1 + answer_2, 1.0);
        assert_eq!(answer_1, correct_1);
        assert_eq!(answer_2, correct_2);

        

    }


}