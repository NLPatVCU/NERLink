#############################################################################
#                                                                           #
#  Trains NERLink's Concept Linking model over the BioCreative 2021 -       #
#  Track 2 data-set and evaluates over the test-set.                        #
#                                                                           #
#############################################################################

import bioc, sys
import numpy as np

sys.path.insert( 0, "../" )

from NERLink import NERLink

# Model/Data Paths
model_save_path = "./concept_linking_bc5cdr_model_ep_1000"

embedding_path  = "./../vectors/chemical_vectors.bin"
concept_id_path = "../data/raw/unique_concept_ids"
write_file_path = "./bioc_formatted_file.xml"
# embedding_path  = ""

# BC7T2 Data-Sets
train_file_path = "../data/raw/BC7T2-CDR-corpus-train.BioC.xml"
val_file_path   = "../data/raw/BC7T2-CDR-corpus-dev.BioC.xml"
eval_file_path  = "../data/raw/BC7T2-CDR-corpus-test.BioC.xml"

# BC5CDR Data-Sets
train_file_path = "../data/raw/CDR_TrainingSet.BioC.xml"
val_file_path   = "../data/raw/CDR_DevelopmentSet.BioC.xml"
eval_file_path  = "../data/raw/CDR_TestSet.BioC.xml"

# NCBI-Disease Data-Sets
train_file_path   = "./../data/raw/NCBItrainset_corpus.xml"
val_file_path     = "./../data/raw/NCBIdevelopset_corpus.xml"
eval_file_path    = "./../data/raw/NCBItestset_corpus.xml"

# Development Data-set (Subset Of BC7T2-CDR Data)
train_file_path = "../data/raw/BC7T2-CDR-corpus-train.BioC.mini.xml"
val_file_path   = "../data/raw/BC7T2-CDR-corpus-test.BioC.mini.xml"
eval_file_path  = "../data/raw/BC7T2-CDR-corpus-test.BioC.mini.xml"

# Model Training Settings
print_debug_log            = False
write_debug_log_file       = False
epochs                     = 50
batch_size                 = 512
learning_rate              = 1e-5
train_weights              = False
use_csr_format             = True
lowercase                  = True
skip_composite_mentions    = True
skip_individual_mentions   = False
restrict_term_context      = False     # True: Restricts Term Sequence Context Two Before And After Sequence Of Interest, False: Maximizes 512 Token Context Space
term_sequence_only         = False     # True: Only Encodes Term Sequence (Removed Context Sequences), False: Include Context Sequences
device_name                = "/gpu:0"
labels_to_skip             = []        # Example: ["chemical"] or ["disease"] Using BC5CDR Data-set
enable_early_stopping      = False
early_stopping_patience    = 2
early_stopping_monitor     = "val_loss"
verbose                    = 2

# Set Loss Function And Activation Function According To 'skip_composite_mention' and 'skip_individual_mention' Settings
if skip_composite_mentions and not skip_individual_mentions:
    loss_function            = "categorical_crossentropy"
    activation_function      = "softmax"
elif skip_individual_mentions and not skip_composite_mentions:
    loss_function            = "binary_crossentropy"
    activation_function      = "sigmoid"
else:
    print( "Error: Invalid 'loss_function' and 'activation_function' Settings" )
    print( "       Both 'skip_composite_mentions' and 'skip_individual_mentions' Cannot Be Concurrently 'True' Or 'False'" )
    exit()

# Initialize NERLink Model Object
nerlink = NERLink( network_model = "concept_linking", model_path = "./../dmis_lab_biobert_base_cased_v1.2",
                   margin = 30, scale = 0.05, use_csr_format = use_csr_format, lowercase = lowercase,
                   trainable_weights = train_weights, loss_function = loss_function, device_name = device_name,
                   enable_early_stopping = enable_early_stopping, early_stopping_patience = early_stopping_patience,
                   early_stopping_monitor = early_stopping_monitor, activation_function = activation_function,
                   verbose = verbose, print_debug_log = print_debug_log, write_log_to_file = write_debug_log_file,
                   skip_composite_mention = skip_composite_mentions, skip_individual_mention = skip_individual_mentions,
                   embedding_a_path = embedding_path, ignore_label_type_list = labels_to_skip )

# Load Unique Concept IDs From File
# nerlink.Get_Data_Loader().Load_Concept_ID_Data( concept_id_path, lowercase = lowercase )

# Read And Encode Our Three Data-sets
if not nerlink.Prepare_Model_Data( data_file_path = train_file_path, val_file_path = val_file_path,
                                   eval_file_path = eval_file_path, lowercase = lowercase ):
    print( "Error Preparing Training, Validation Or Evaluation Data" )
    exit()

# Print Size Out Model Output Space
print( "Number Of Outputs: " + str( len( nerlink.Get_Data_Loader().Get_Concept_ID_Dictionary() ) ) )

# Get Encoded Data-Sets
train_input  = nerlink.Get_Data_Loader().Get_Concept_Inputs()
train_output = nerlink.Get_Data_Loader().Get_Concept_Outputs()
dev_input    = nerlink.Get_Data_Loader().Get_Concept_Validation_Inputs()
dev_output   = nerlink.Get_Data_Loader().Get_Concept_Validation_Outputs()
test_input   = nerlink.Get_Data_Loader().Get_Concept_Evaluation_Inputs()
test_output  = nerlink.Get_Data_Loader().Get_Concept_Evaluation_Outputs()

# Check(s)
if train_input is None or train_output is None:
    print( "Error: Train Inputs Or Outputs == None" )
    exit()
elif dev_input is None or dev_output is None:
    print( "Error: Dev Inputs Or Outputs == None" )
elif test_input is None or test_output is None:
    print( "Error: Test Inputs Or Outputs == None" )
    exit()

# Set Class Weights
#   Let's Just Use A Basic Weighting Scheme To Scale Down 'CUI-LESS' Representation Class Imbalance
cui_less_token       = nerlink.Get_Data_Loader().Get_CUI_LESS_Token()
cui_less_concept_idx = nerlink.Get_Data_Loader().Get_Concept_ID( cui_less_token )
class_weights        = { idx : 1 for idx in range( nerlink.Get_Data_Loader().Get_Number_Of_Unique_Concepts() ) }
class_weights[cui_less_concept_idx] = 0.125

# Train The Concept Linking Model
nerlink.Fit( encoded_input = train_input, encoded_output = train_output,
             val_encoded_input = dev_input, val_encoded_output = dev_output,
             epochs = epochs, learning_rate = learning_rate, batch_size = batch_size,
             class_weights = class_weights )

# Save The Trained Model
nerlink.Save_Model( model_save_path )

########################################
#            EVALUATION                #
########################################

print( "\n**************************" )
print( "*** EVALUATION METRICS ***"   )
print( "**************************\n" )

metrics = nerlink.Evaluate( encoded_input = test_input, encoded_output = test_output )
print( "Metrics: " + str( metrics ) )

eval_per_class_metrics = nerlink.Evaluate_Manual( encoded_input = test_input, encoded_output = test_output )

# Print Avg Metrics Per Class
overall_metrics = eval_per_class_metrics["Avg / Total"]

for metric, value in overall_metrics.items():
    print( str( metric ) + " :  " + str( value ) )

print( "Accuracy (Data) : " + str( eval_per_class_metrics["Accuracy (Data)"] ) )
print( "Accuracy (Model): " + str( eval_per_class_metrics["Accuracy (Model)"] ) )

##################################################
# Print BioCreative Formatted Data Instance File #
##################################################

# Open BioC Read And Write File Handles
reader = bioc.BioCXMLDocumentReader( eval_file_path )
writer = bioc.BioCXMLDocumentWriter( write_file_path )

# Write Collection Info
collection_info = reader.get_collection_info()
writer.write_collection_info( collection_info )

# Compose A Concept Index To Concept ID List
concept_id_list = list( nerlink.Get_Data_Loader().Get_Concept_ID_Dictionary().keys() )

# Let's Capitalize All Concept ID Elemnets In The List
concept_id_list = [ concept_id.upper() for concept_id in concept_id_list ]

# Process Each Document In The File
for document in reader:
    # NER/CL - Iterate Through Each Passage In The Given Document
    for passage in document.passages:
        for annotation in passage.annotations:
            # Get Annotation Type
            if "identifier" in annotation.infons: annotation_type = "identifier"
            if "MESH"       in annotation.infons: annotation_type = "MESH"

            if annotation_type is None:
                print( "Error: Annotation Type Not Implemented" )
                raise NotImplementedError

            # Set Term Concept Label As CUI-Less (Default Label)
            annotation.infons[annotation_type] = "-"

            # Skip Annotation Text As 'None'
            if not annotation.text: continue

            annotation_text   = annotation.text.lower() if lowercase else annotation.text
            annotation_id     = nerlink.Get_Data_Loader().Get_Token_ID( annotation_text )
            annotation_type   = None

            concept_id_labels = []

            # Set All Concept IDs That Do Not Exist In The Dictionary As False Negative Predictions
            if annotation_id == -1:
                concept_id_labels = []
            else:
                concept_id_labels = nerlink.Predict( encoded_input = [annotation_id] )

            # Determine Number Of Prediction Instance Dimensions ( We Only Want Two i.e. shape = ( num_instances, concept_id_list_size ) )
            concept_id_labels = np.asarray( concept_id_labels, np.object )

            if concept_id_labels.ndim == 3: concept_id_labels = concept_id_labels[0][0]
            if concept_id_labels.ndim == 2: concept_id_labels = concept_id_labels[0]

            concept_id_labels = list( concept_id_labels )

            # Adjust For CUI-LESS Token i.e. Set '<*>CUI-LESS<*>' To '-' To Represent CUI-Less Concept ID
            concept_id_labels = [ "-" if concept_id == nerlink.Get_Data_Loader().Get_CUI_LESS_Token() else concept_id for concept_id in concept_id_labels ]

            # Assign The Concept ID Labels For The Given Annotation Instance (Entry Term)
            concept_id_labels = [ concept_id.upper() for concept_id in concept_id_labels ]
            annotation.infons[annotation_type] = ",".join( concept_id_labels ) if len( concept_id_labels ) > 0 else "-"

    writer.write_document( document )

# Close BioC XML Writer
writer.close()

print( "~Fin" )