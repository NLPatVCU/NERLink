#############################################################################
#                                                                           #
#  Trains NERLink's BERT model and evaluates over the test data-set.        #
#                                                                           #
#############################################################################

import sys

sys.path.insert( 0, "../" )

from NERLink import NERLink

# Begin Code

# Model/Data Paths
model_path      = "./bert_nlmchem_model_ep_30"

train_file_path = "../data/raw/BC7T2-CDR-corpus-train.BioC.xml"
val_file_path   = "../data/raw/BC7T2-CDR-corpus-dev.BioC.xml"
eval_file_path  = "../data/raw/BC7T2-CDR-corpus-test.BioC.xml"

# Development Data-set (Subset Of CDR Data)
train_file_path = "../data/raw/BC7T2-CDR-corpus-train.BioC.mini.xml"
val_file_path   = "../data/raw/BC7T2-CDR-corpus-test.BioC.mini.xml"
eval_file_path  = "../data/raw/BC7T2-CDR-corpus-test.BioC.mini.xml"

# Model Training Settings
epochs                   = 30
batch_size               = 1
learning_rate            = 1e-5
train_bert_layer_weights = True
lowercase                = False
device_name              = "/gpu:0"

nerlink = NERLink( network_model = "ner_bert", model_path = "./../dmis_lab_biobert_base_cased_v1.2",     # 'model_path' Default: bert-base-uncased
                   loss_function = "sparse_categorical_crossentropy", activation_function = "softmax",
                   trainable_weights = train_bert_layer_weights, device_name = device_name, verbose = 2 )

# Read And Encode Our Three Data-sets
if not nerlink.Prepare_Model_Data( data_file_path = train_file_path, val_file_path = val_file_path,
                                   eval_file_path = eval_file_path, lowercase = lowercase ):
    print( "Error Preparing Training, Validation Or Evaluation Data" )
    exit()

# Get Encoded Data-Sets
train_input  = nerlink.Get_Data_Loader().Get_NER_Inputs()
train_output = nerlink.Get_Data_Loader().Get_NER_Outputs()
dev_input    = nerlink.Get_Data_Loader().Get_NER_Validation_Inputs()
dev_output   = nerlink.Get_Data_Loader().Get_NER_Validation_Outputs()
test_input   = nerlink.Get_Data_Loader().Get_NER_Evaluation_Inputs()
test_output  = nerlink.Get_Data_Loader().Get_NER_Evaluation_Outputs()

# Check(s)
if train_input is None or train_output is None:
    print( "Error: Train Inputs Or Outputs == None" )
    exit()
elif dev_input is None or dev_output is None:
    print( "Error: Dev Inputs Or Outputs == None" )
elif test_input is None or test_output is None:
    print( "Error: Test Inputs Or Outputs == None" )
    exit()

# Train The NER Model
nerlink.Fit( encoded_input = train_input, encoded_output = train_output,
             val_encoded_input = dev_input, val_encoded_output = dev_output,
             epochs = epochs, learning_rate = learning_rate, batch_size = batch_size )

# Save The Trained Model
# nerlink.Save_Model( model_path )

########################################
#            EVALUATION                #
########################################

print( "\n**************************" )
print( "*** EVALUATION METRICS ***" )
print( "**************************\n" )

eval_per_class_metrics = nerlink.Evaluate_Manual( encoded_input = test_input, encoded_output = test_output )

# Print Avg Metrics Per Class
overall_metrics = eval_per_class_metrics["Avg / Total"]

for metric, value in overall_metrics.items():
    print( str( metric ) + " :  " + str( value ) )

print( "Accuracy: " + str( eval_per_class_metrics["Accuracy (Data)"] ) )

print( "~Fin" )