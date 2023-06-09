use std::{io::Read, array};

use ndarray::*;



#[warn(dead_code)]
struct FeedForwardNetwork {
    epoch: u32,

}


impl FeedForwardNetwork {

    // create a new ffn object
    fn new() -> FeedForwardNetwork {
        FeedForwardNetwork { epoch: 0 }
    }

    fn train() {
        todo!()
    }

    fn test() {

    }


}



//TODO: need a function to read in data and convert it to an array.


fn read_training_labels() -> Array1<f64> {

    todo!()
}

fn read_training_data() -> Array<f64, Ix> {
    todo!()
}



fn main() -> std::io::Result<()> {
    FeedForwardNetwork::new();

    // let data: Vec<u8> = fs::read("mnist_data/train-labels-idx1-ubyte")?;

    // let test2: &[u8; 4] = &data[3..7].try_into().unwrap();

    // let test: [u8; 4] = [0, 0, 234, 96];

    // let n: u32 = u32::from_be_bytes(*test2);

    // print!("{}", n);

    


    let mut file = std::fs::File::open("mnist_data/train-labels-idx1-ubyte")?;

    let mut buffer: Vec<u8> = Vec::new();

    file.read_to_end(&mut buffer);

    //let mut byte_vec: Vec<u8> = vec![0; buffer.len()];


    // let mut i: usize = 0;

    // for num in buffer.iter() {
    //     byte_vec[i] = u8::to_le_bytes(*num); 
    //     i+=1;
    // }    

    

    let decimal_32: &[u8; 4] = &buffer[4..8].try_into().unwrap();
    
    let binary_32: u32 = u32::from_be_bytes(*decimal_32);

   


    println!("{:?}", decimal_32);


    println!("{}", binary_32);

    let n = buffer.len();

    let data: Vec<u8>  = buffer[8..n].try_into().unwrap();


    let arr = Array::from(data);

    let arr = arr.into_shape((2, 30000)).unwrap();
        
    
    
    println!("{:?}", arr);


    Ok(())
} 
