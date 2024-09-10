import argparse
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the preprocessed data.", type=str, required=True)
    parser.add_argument("--val_size", help="Size (int) of the validation dataset.", type=int, required=True)
    parser.add_argument("--test_size", help="Size (int) of the test dataset.", type=int, required=True)
    parser.add_argument("--save_dir", help="Path to save the data splits.", type=str, default='data/')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load the data
    df = pd.read_csv(args.data_path)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the data
    test_df = df.iloc[:args.test_size]
    val_df = df.iloc[args.test_size:args.test_size+args.val_size]
    train_df = df.iloc[args.test_size+args.val_size:]
    
    # Print the number of samples per dataset
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save the splits
    train_df.to_csv(os.path.join(args.save_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(args.save_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(args.save_dir, 'test.csv'), index=False)
    
    print(f"Data splits saved in {args.save_dir}")


if __name__ == '__main__':
    main()