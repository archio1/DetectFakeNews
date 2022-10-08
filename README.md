# Fake news detection software
## About program 

The program gives you opportunity to detect true news or not. 

## Requirements
- Python => 3.8
- LFS track for big files (resources folder) 

## Start program

The program is run from a file: `start_program_of_detect_fake.py`
The program has 4 modes of work:    
- create passive-aggressive model   
- check news by passive-aggressive model 
- create Long Short-Term Memory network, check news by LSTM
- check news by LSTM model

Commands for every mode from console below:
```
python start_program_of_detect_fake.py --create passive_aggressive 'way to the file for training' 
python start_program_of_detect_fake.py --check passive_aggressive 'way to the file for test' 
python start_program_of_detect_fake.py --create neural_network 'way to the file for training'
python start_program_of_detect_fake.py --check neurual_network 'way to the file for test' 
```

## Testing

The tests were written by PyTest framework. The command to start tests:
    
`pytest tests`

