import argparse
from utility import *

def main(args: argparse.Namespace):
    '''
    Main function for image prediction using saved model.
    
    Args:
        - args (argparse.Namespace): Parsed command-line arguments
        
    Returns:
        - None
    '''
    
    # Load pre-trained model
    model = load_model(args.model_filepath)
    
    if args.verbose:
        # Show model summary
        model.summary()
    
    # Load and preprocess image
    image = load_and_preprocess_image(args.image_path)
    
    # Make predictions using pre-trained model
    top_probs, top_classes = predict(image, model, args.top_k)
    
    if args.category_names:
        category_names, first_key = load_category_names(args.category_names)
        # Map numbered classes to category names
        # If category names start counting by 1 or greater, indices must be adjusted
        top_classes = [category_names[str(value+first_key)] for value in top_classes]
    
    if args.verbose:
        # Print command-line arguments
        for arg_name, arg_value in vars(args).items():
            print(f"{arg_name}='{arg_value}'")
    
    # Print top_k predictions
    print("-----")
    print(f"Top {args.top_k} predictions:")
    for class_name, probability in zip(top_classes, top_probs):
        print(f"{class_name}: {probability:.4f}")
    
if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Flower Image Classifier - Image prediction using a saved model")
    
    # Define command-line arguments
    parser.add_argument("image_path", help="Path to the image to predict")
    parser.add_argument("model_filepath", help="Path to the saved model")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Return the to K most likely classes")
    parser.add_argument("--category_names", "-c", help="Path to JSON file mapping labels to flower names")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose information")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)