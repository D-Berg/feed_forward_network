use std::{io::Read, sync::Arc};
use ndarray::*;
use std::io::Write;
use Layer::*;

/// A Feed Forward Network to be used on mnist data
/// Input a 28*28 = 784.
/// output a 10 length vector of probs 
/// have a function that take the index of max element and return that to 
/// get classification.
#[derive(Debug)]
struct FeedForwardNetwork {
    layers: Vec<Layer>,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,

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
    pub fn train(epochs: u32) {
        todo!()
    }

    pub fn test() {

    }

    /// adds a layer to to the network
    pub fn add_layer(mut self, layer: Layer) -> Self {
        // TODO: need to update size of weights and biases tensors

        self.layers.push(layer);

        match layer {
            ReLU(n_nuerons) => {

                self.add_weight_and_bias_dim(n_nuerons);
                
            },

            Flatten(_) => { // assume it is always input
                
                
            },

            SoftMax(n_nuerons) => {

                self.add_weight_and_bias_dim(n_nuerons);

            }
            
        }

        return self;

    }

    fn add_weight_and_bias_dim(&mut self, n_nuerons: usize) -> &Self {


        let n_layers = self.layers.len();

        // -2 since indexing starts on 0
        let n_nuerons_last_layer: usize = self.layers[n_layers - 2].get_n_neurons();


        // update weights dim
        self.weights.push(Array2::<f64>::zeros((n_nuerons, n_nuerons_last_layer)));

        return self;

    }


}

#[derive(Debug, Clone, Copy)]
enum Layer{
    Flatten(usize),
    ReLU(usize),
    SoftMax(usize), 
} 

impl Layer {
    fn get_n_neurons(&self) -> usize {

        match self {
            Flatten(val) => *val,
            ReLU(val) => *val,
            SoftMax(val) => *val,
        }
        
    }
}



/// Function to be used by get data functions 
/// to minimize code repetition.
fn read_in_data(file_name: &str) -> Vec<u8> {

    let mut file = std::fs::File::open(file_name).expect("Error opening file");

    let mut buffer: Vec<u8> = Vec::new();

    file.read_to_end(&mut buffer).expect("Error reading in file");

    return buffer;
}


//TODO: need a function to read in data and convert it to an array.

/// Extracts data useful data from read_in_data and returns training labels as a vector.
fn get_training_labels() -> ArrayBase<OwnedRepr<u8>, Dim<[usize; 1]>> {

    // we get unclean data from reading the file
    // we need to extract the useful information 

    let file_name: &str = "mnist_data/train_set_labels";

    let un_clean_data: Vec<u8> = read_in_data(file_name);

    //length of unclean data
    let n: usize = un_clean_data.len();

    let clean_data: Vec<u8>  = un_clean_data[8..n].try_into().unwrap();

    let data_array: ArrayBase<OwnedRepr<u8>, Dim<[usize; 1]>> = Array::from(clean_data);

    return data_array;

}

fn get_training_images() -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>> {
    
    let file_name: &str = "mnist_data/train_set_images";

    let un_clean_data: Vec<u8> = read_in_data(file_name);


    dbg!(&un_clean_data[0..4]);

    let n: usize = un_clean_data.len();

    let n_images_binary: &[u8; 4] = &un_clean_data[4..8].try_into().unwrap();
    
    let n_images: usize = u32::from_be_bytes(*n_images_binary) as usize;
    dbg!(n_images);

    let n_rows: usize = un_clean_data[11] as usize;
    dbg!(n_rows);

    let n_cols: usize = un_clean_data[15] as usize;
    dbg!(n_cols);


    let clean_data: Vec<u8> = un_clean_data[16..n].try_into().unwrap();

    dbg!(clean_data.len() as f64 /  784.0);


    let mut clean_data_f64: Vec<f64> = vec![0.0; clean_data.len()];

    // convert to f64
    for i in 0..clean_data.len() {
        clean_data_f64[i] = clean_data[i] as f64;
    }

    // convert to array

    let data_array: Array3<f64> = Array::from_shape_vec((n_images, n_rows, n_cols) ,clean_data_f64).expect("Error: wrong shape");


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


    let stdout = std::io::stdout();
    let mut handle = std::io::BufWriter::new(stdout);
    
    for i in 0..n_rows {
        for j in 0..n_cols {


            let pix_val: u8 = image_array[[i, j]] as u8;

            // print!("{}", pixel.truecolor(pix_val, pix_val, pix_val));

            write!(handle, "\x1b[48;2;{};{};{}m  ", pix_val, pix_val, pix_val).expect("failed to write to handle");

        }

        write!(handle, "\x1b[0m\n").expect("failed to write to handle");
    }

    handle.write_all(b"\n").expect("failed to write out all");


}


fn main() {

    let training_labels = get_training_labels();


    let training_images = get_training_images();

    
    for i in 0..5 {
        print_picture(i, &training_images, &training_labels);
    }

    let network: FeedForwardNetwork = FeedForwardNetwork::new()
        .add_layer(Flatten(2))
        .add_layer(ReLU(4))
        .add_layer(SoftMax(3));

    for weight in network.weights.iter() {
        println!("{:?}", weight);
    }
        

    // let test: Vec<f64> = vec![0.0; 4];

    // //create dynamic array
    // let mut weigths: ArrayD<f64> = ArrayD::<f64>::zeros(IxDyn(&[0]));

    // println!("{:?}", weigths);

    // weigths = ArrayD::<f64>::zeros(IxDyn(&[]));

    // println!("{:?}", weigths);
    

} 
