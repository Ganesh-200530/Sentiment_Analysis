import os
import sys

def main():
    print("--- Sentiment Analysis Project Runner ---")
    print("1. Train Model (src/train.py)")
    print("2. Predict Sentiment (src/predict.py)")
    
    choice = input("\nEnter choice (1 or 2): ")
    
    if choice == '1':
        print("\nStarting training...")
        os.system('python src/train.py')
    elif choice == '2':
        print("\nStarting prediction interface...")
        os.system('python src/predict.py')
    else:
        print("Invalid choice. Please run the script again and select 1 or 2.")

if __name__ == "__main__":
    main()
