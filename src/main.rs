use std::fmt::format;
// use std::io::Write;
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
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};


// blas
extern crate blas_src;


// threading 
use std::thread;
use std::thread::JoinHandle;
use std::sync::{Arc, Mutex, MutexGuard};





/// A Feed Forward Network to be used on mnist data
/// Input a 28*28 = 784.
/// output a 10 length vector of probs 
/// have a function that take the index of max element and return that to 
/// get classification.
#[derive(Debug, Clone)]
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

        let loss_function_string: &str = match loss_function {
            CrossEntropy => "Cross Entropy",
            LeastSquares => "Least Squares",
        };

        println!("{}", "Start training...\n".red());

        println!("Using Loss Function: {} \n", loss_function_string.blue());

        let mut total_training_time: f64 = 0.0;

        //learning rate 
        let eta: f64 = 1.0;

        let shape_images: &[usize] = &training_images.shape();

        let n_images: usize = shape_images[0];

        let n_neurons_input_layer: usize = shape_images[1] * shape_images[2];

        self.initialize_weights_and_biases(n_neurons_input_layer);

        

        'epoch_loop: for epoch in 0..epochs {

            let now = std::time::Instant::now();

            println!("{} {}/{}", "Epoch:".blue(), epoch + 1, epochs);
            // std::io::stdout().flush().expect("Failed to flush stdout");
            
            
            // Threading--------------------------------------------------------

            // TODO: Change num_threads to the sytems
            let n_threads: usize = 8;

            //number of images each thread will take care of
            // TODO: change to implement modulo so we dont forget an image
            let thread_batch_size: usize = n_images / n_threads;

            let mut handles: Vec<JoinHandle<()>> = Vec::with_capacity
            (n_threads);



            // DATA ACCESSIBLE BY ALL THREADS-----------------------------------
            // Threads shoud only write to these once to increase perfomance
            // with the exception of maybe progress bar (pb).

            // initialize gradients

            // TODO: Change gradients to vecs of vecs

            let w_grads_full_arc: Arc<Mutex<Vec<Vec<Array2<f64>>>>> = Arc::new(Mutex::new(Vec::with_capacity(n_threads)));

            let b_grads_full_arc: Arc<Mutex<Vec<Vec<Array2<f64>>>>> = Arc::new(Mutex::new(Vec::with_capacity(n_threads)));
            

            let costs_full_arc: Arc<Mutex<Vec<Vec<f64>>>> = Arc::new(Mutex::new(Vec::with_capacity(n_threads)));

            // let pb_arc= Arc::new(Mutex::new(indicatif::ProgressBar::new(n_images as u64)));

            // A vec of vec to minimize locking
            let n_correct_arc: Arc<Mutex<Vec<Vec<u32>>>>  = Arc::new(Mutex::new(Vec::with_capacity(n_threads)));

            // END OF DATA ACCESIBLE BY ALL THREADS-----------------------------

            // Progress bar-------------------------------------

            let multi_bar = MultiProgress::new();
            
            let sty = ProgressStyle::with_template(
                "{elapsed} {bar:50.yellow} {pos:>7}/{len:5} {msg:10.yellow}",
            )
                .unwrap()
                .progress_chars("##-");

                let sty_finished = ProgressStyle::with_template(
                    "{elapsed} {bar:50.green} {pos:>7}/{len:5} {msg:10.green}",
                )
                    .unwrap()
                    .progress_chars("##-");

            let mut p_bars: Vec<ProgressBar> = Vec::with_capacity(thread_batch_size);

            for thread_id in 0..n_threads {
                let pb = ProgressBar::new(thread_batch_size as u64);

                pb.set_style(sty.clone());

                p_bars.push(multi_bar.insert(thread_id, pb));
                
            }
            
    
    
            for thread_id in 0..n_threads {


                // Think clones only live as long as its thread do
                let mut self_clone: FeedForwardNetwork = self.clone();
                let loss_function_clone: LossFunction = loss_function.clone();

                let  w_grads_full_arc = Arc::clone(&w_grads_full_arc);

                let  b_grads_full_arc = Arc::clone(&b_grads_full_arc);

                let n_correct_arc = Arc::clone(&n_correct_arc);


                let costs_full_arc = Arc::clone(&costs_full_arc);


                
                // Gives name to thread
                let thread_builder: thread::Builder = thread::Builder::new().name(format!("Thread {}", thread_id));

                let start_index: usize = thread_id * thread_batch_size;

                let end_index: usize = (thread_id + 1) * thread_batch_size;


                let training_images_batch: Array3<f64> = training_images.slice(s![start_index..end_index, .., ..]).to_owned();

                let training_labels_batch: Array1<u8> = training_labels.slice(s![start_index..end_index]).to_owned();



                let multi_bar_clone: MultiProgress = multi_bar.clone();


                let p_bars_clone: Vec<ProgressBar> = p_bars.clone();

                let pb = p_bars_clone[thread_id].clone();

                let sty_finished_clone = sty_finished.clone();



                let handle: JoinHandle<()> = thread_builder.spawn( move || {

                    let (mut w_grads, mut b_grads) = self_clone.create_gradients();

                    pb.set_position(thread_id as u64);

                    // Creation of data vecs that shall be bushed to the big one
                    // local for thread 

                    let mut n_correct_local: Vec<u32> = Vec::with_capacity(thread_batch_size);

                    let mut costs_local: Vec<f64> = Vec::with_capacity(thread_batch_size);

                    // Start of training per thread-----------------------------

                    for i in 0..thread_batch_size {



                        // data to be trained on in this thread
                        let image: Array2<f64> = training_images_batch.slice(s![i, .., ..]).to_owned();
                        let label: u8 = training_labels_batch[[i]];

                        // calculate activations for image 
                        let activations: Vec<Array2<f64>> = self_clone.feed_forward(image);

                         // TODO: Move this to its own function
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
                            n_correct_local.push(1);
                        }


                        // backpropegate image in network to get deltas
                        let deltas: Vec<Array2<f64>> = self_clone.backpropegate(
                            &activations, 
                            &label, 
                            &loss_function_clone
                        );

                        // calculate gradient for dC_x/dW and dC_x/db (matrix form)
                        let L = self_clone.layers.len();

                        for l in (0..L).rev() {

                            let a_l_min_1: &Array2<f64> = &activations[l];
                            
                            
                            let weight_grad_x: Array2<f64> = deltas[l].dot(&a_l_min_1.t());

                            let bias_grad_x: Array2<f64> = deltas[l].to_owned();
                            
        
                            w_grads[l] = &w_grads[l] + weight_grad_x;
        
                            b_grads[l] = &b_grads[l] + bias_grad_x;
                            
        
                        }


                        // creates vec [0, 0, 0 , 1, ..., 0]
                        let mut label_vec: Vec<f64> = vec![0.0; 10]; 

                        label_vec[label as usize] = 1.0;

                        let y: Array2<f64> = Array2::<f64>::from_shape_vec((label_vec.len(), 1), label_vec).expect("failed to create y");
                    
                        let a_L = &activations[activations.len()-1];


                        let cost: f64 = match loss_function_clone {
                            LeastSquares => {
    
                                (a_L - y).mapv(|a: f64| a.pow(2)).sum()
                            
                            },
                            CrossEntropy =>{
    
    
    
                             -a_L[[label as usize, 0]].ln()
                            },
                        };

                        costs_local.push(cost);

                        
                        pb.set_message(format!("{} training", thread::current().name().unwrap()));
                        pb.inc(1);
                        


                    } // end of for i

                    

                    // Lock and Push all data here------------------------------
                    let mut n_correct_full = n_correct_arc.lock().expect("Failed to lock n_correct");

                    n_correct_full.push(n_correct_local);


                    let mut costs_full = costs_full_arc.lock().expect("Failed to lock costs");

                    costs_full.push(costs_local);

                    //push local gradients of thread(id) 
                    let mut w_grads_full = w_grads_full_arc.lock().unwrap();
                    w_grads_full.push(w_grads);

                    let mut b_grads_full = b_grads_full_arc.lock().unwrap();
                    b_grads_full.push(b_grads);

                    // let pb = pb_arc.lock().unwrap();

                    // pb.inc(thread_batch_size as u64);

                    pb.set_style(sty_finished_clone);
                    pb.tick();
                    pb.finish_with_message(format!("{} finnished", thread::current().name().unwrap()));

                }).unwrap();

                handles.push(handle);
                

            } // end of thread creation

            // join all threads
            for handle in handles {
                handle.join().unwrap();
            }

            // multi_bar.clear().unwrap();


            // Update weights W^l = W^l - grad(W^l)
            // remember grad(W^l) = mean(dC_x/dW^l)

            // Vector of gradients from each thread
            let w_grads_from_threads = w_grads_full_arc.lock().expect("Failed to lock w_grads");

            let b_grads_from_threads = b_grads_full_arc.lock().expect("Failed to lock b_grads");

            let (mut weight_gradients, mut bias_gradients) = self.create_gradients();

            for thread_id in 0..n_threads {

                let w_grads_from_thread = &w_grads_from_threads[thread_id];

                let b_grads_from_thread = &b_grads_from_threads[thread_id];

                for l in 0..weight_gradients.len() {

                    weight_gradients[l] = &weight_gradients[l] + &w_grads_from_thread[l];

                    bias_gradients[l] = &bias_gradients[l] + &b_grads_from_thread[l];

                }

            }

            

            self.sgd(&weight_gradients, &bias_gradients, n_images, &eta);


            
            // Update weights b^l = b^l - grad(b^l)
            // grad(b^l) = mean(dC_x/db^l) = mean(delta^l_x)

            let n_correct_vec_vec = n_correct_arc.lock().unwrap();

            // let n_correct = &*n_correct;

            let mut n_correct: u32 = 0;

            for n_correct_vec in n_correct_vec_vec.iter() {
                let n: u32 = n_correct_vec.iter().sum();
                n_correct += n;
            }    

            let accuracy: f64 = (n_correct) as f64 / n_images as f64;

            let mut total_cost = 0.0;
            

            let costs_full = costs_full_arc.lock().unwrap();

            for costs in costs_full.iter() {

                let cost: f64 = costs.iter().sum();
                total_cost += cost;

            }
            

            let loss: f64 = total_cost / n_images as f64;


            let epoch_time: f64 =  now.elapsed().as_secs_f64();

            total_training_time += epoch_time;

            println!(
                "{} {:.2}s, {} {:.2}, {}: {:.2}", 
                "Time:".blue(),
                epoch_time,
                "Accuracy:".blue(),
                accuracy,
                "Loss".blue(),
                loss
            );

            // total_training_time += pb.elapsed();
            
            print!("\n");

        } // end of epoch loop


        println!("\n{} {:.2}", "Finished training in:".green(), total_training_time);

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

    /// Create gradients. 
    /// Creates W_grad a vector of W^l_grad
    fn create_gradients(&self) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {


        let mut weight_gradients: Vec<Array2<f64>> = Vec::with_capacity(self.weights.len());
            
        let mut bias_gradients: Vec<Array2<f64>> = Vec::with_capacity(self.weights.len());

        for l in 0..self.weights.len() {

            let weight_shape: &[usize] = &self.weights[l].shape();
            let bias_shape = &self.biases[l].shape();

            weight_gradients.push(Array2::<f64>::zeros([weight_shape[0], weight_shape[1]]));

            bias_gradients.push(Array2::<f64>::zeros([bias_shape[0], bias_shape[1]]));

        }

        return (weight_gradients, bias_gradients);

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

#[derive(Clone)]
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