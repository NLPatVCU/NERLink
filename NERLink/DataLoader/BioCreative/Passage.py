#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Named Entity Recognition + Concept Linking Package                                    #
#    --------------------------------------------------                                    #
#                                                                                          #
#    Date:    05/28/2020                                                                   #
#    Revised: 10/02/2021                                                                   #
#                                                                                          #
#    BioCreative Data Loader Passage Class For The NERLink Package.                        #
#                                                                                          #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################


############################################################################################
#                                                                                          #
#    Passage Class                                                                         #
#                                                                                          #
############################################################################################

class Passage( object ):
    def __init__( self, document_id = -1, passage = "", passage_original = "", passage_type = "",
                  annotations = [], annotation_labels = [], annotation_indices = [],
                  annotation_concept_ids = [], concepts = {}, concept_linking = {} ):
        self.document_id             = document_id
        self.passage                 = passage                 # Sequence Of Text
        self.passage_original        = passage_original        # Sequence Of Text (Original)
        self.passage_type            = passage_type            # Passage Type i.e. Title, Abstract, etc.
        self.annotations             = annotations             # List Of Entity Tokens
        self.annotation_labels       = annotation_labels       # List Of Entity Types
        self.annotation_indices      = annotation_indices      # List Of Entity Token Indices : <:> Separates Indexing Instances
        self.annotation_concept_ids  = annotation_concept_ids  # List Of Entity Token Concept ID Mappings
        self.concepts                = concepts                # Chemical Indexing - Key: Term => Value: Concept Identifier
        self.concept_linking         = concept_linking         # Concept Linking - Key: MeSH ID => Value: Term (Do Not Use - MeSH ID May Have > 1 Term)
        self.composite_mention_list  = []
        self.individual_mention_list = []

    # Accessors
    def Get_Document_ID( self ):                    return self.document_id
    def Get_Passage( self ):                        return self.passage
    def Get_Passage_Original( self ):               return self.passage_original
    def Get_Passage_Type( self ):                   return self.passage_type
    def Get_Annotations( self ):                    return self.annotations.copy()
    def Get_Annotation_Labels( self ):              return self.annotation_labels.copy()
    def Get_Annotation_Indices( self ):             return self.annotation_indices.copy()
    def Get_Annotation_Concept_IDs( self ):         return self.annotation_concept_ids.copy()
    def Get_Concepts( self ):                       return self.concepts.copy()
    def Get_Concept_Linking( self ):                return self.concept_linking.copy()
    def Get_Composite_Mention_List( self ):         return self.composite_mention_list.copy()
    def Get_Individual_Mention_List( self ):        return self.individual_mention_list.copy()

    # Mutators
    def Set_Document_ID( self, value ):             self.document_id = value
    def Set_Passage( self, passage ):               self.passage = passage
    def Set_Passage_Original( self, passage ):      self.passage_original = passage
    def Set_Passage_Type( self, passage_type ):     self.passage_type = passage_type
    def Set_Annotations( self, annotations ):       self.annotations = annotations
    def Set_Annotation_Labels( self, labels ):      self.annotation_labels = labels
    def Set_Annotation_Indices( self, indices ):    self.annotation_indices = indices
    def Set_Annotation_Concept_IDs( self, ids ):    self.annotation_concept_ids = ids
    def Set_Concepts( self, concepts ):             self.concepts = concepts
    def Set_Concept_Linking( self, dict ):          self.concept_linking = dict
    def Set_Composite_Mention_List( self, list ):   self.composite_mention_list = list
    def Set_Individual_Mention_List( self, list ):  self.individual_mention_list = list
