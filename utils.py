import numpy as np
import geoopt
import torch



class MDADataset(torch.utils.data.Dataset):
    """ 
        Pytorch Dataset for Metabolite-Disease Associations 
    
        Attributes
        ----------
        associations : 1D array
            Binary associations between the corresponding metabolites and diseases
        metabolites : 1D array
            Metabolite indices
        diseases : 1D array
            Disease indices

        Methods
        -------
        init(metabolites, diseases, associations)
            Store the given associations as well as the metabolite and disease indices
        len()
            Return the number of stored associations
        getitem(idx)
            Return the associations given by idx and the corresponding metabolite and disease indices
    """
    def __init__(self, metabolites: np.array, diseases: np.array, associations: np.array) -> None:
        super().__init__()
        self.metabolites = metabolites      
        self.diseases = diseases
        self.associations = associations
        
    def __len__(self) -> int:
        return(self.associations.shape[0])

    def __getitem__(self, idx: int) -> dict:
        return {
            "metabolite": self.metabolites[idx],
            "disease": self.diseases[idx],
            "association": self.associations[idx]
        }
    
def sample_negative_associations(matrix, negative_sample_ratio):
    """
        Random sampling of negative associations

        Parameters
        ----------
        matrix : sparse matrix
            Input association matrix containing only zeros and ones
        negative_sample_ratio: int
            Determines how many negative associations to sample for each known positive one

        Returns
        ------
        sampled_matrix: sparse matrix
            Output association matrix with positive (1), negative (-1), and unknown (0) associations
    """
    zero_row, zero_col = np.where(matrix.toarray() == 0)
    sampled_ids = np.random.choice(np.arange(zero_row.shape[0]), matrix.sum()*negative_sample_ratio, replace = False)
    sampled_matrix = matrix.copy()
    sampled_matrix[zero_row[sampled_ids], zero_col[sampled_ids]] = -1
    return sampled_matrix

def unfold_association_matrix(matrix):
    """
        Extract all the positive and negative associations with the corresponding metabolite and disease indices

        Parameters
        ----------
        matrix : sparse matrix
            Input association matrix containing positive (1), negative (-1), and unknown (0) associations
            Rows are corresponding to metabolites, while columns are to diseases

        Returns
        ------
        metabolites : 1D array
            Metabolite indices
        diseases : 1D array
            Disease indices
        associations : 1D array
            Binary associations between the corresponding metabolites and diseases
    """
    metabolites = []
    diseases = []
    associations = []
    for r, row in enumerate(matrix):
        for c, col in enumerate(row.indices):
            metabolites.append(r)
            diseases.append(col)
            associations.append(row.data[c])
    metabolites = np.array(metabolites).astype(int)      
    diseases = np.array(diseases).astype(int)
    associations = (np.array(associations).astype(np.float64) + 1) / 2
    return (metabolites, diseases, associations)



class ManifoldEmbedding(torch.nn.Module):
    """ 
        Pytorch Module for Metabolite and Disease Embeddings
    
        Attributes
        ----------
        embeddings: geoopt.ManifoldParameter
            Model parameters on a given manifold

        Methods
        -------
        init(vocab_size, embed_dim, manifold)
            Initialize the parameters with a normal distribution on the manifold
        forward(x)
            Return xth parameter
    """
    def __init__(self, vocab_size, embed_dim, manifold):
        super(ManifoldEmbedding, self).__init__()

        self.__embeddings = manifold.random_normal((vocab_size, embed_dim), mean = manifold.origin((vocab_size, embed_dim)), std = 1.0)
        self.__embeddings = geoopt.ManifoldTensor(self.__embeddings, manifold=manifold)
        self.__embeddings.proj_()
        self.__embeddings = geoopt.ManifoldParameter(self.__embeddings)

    def forward(self, x):
        return self.__embeddings[x]
    
def process_batch(batch, manifold, device, metabolite_embedding_module, disease_embedding_module):
    """
        Give predictions for a batch

        Parameters
        ----------
        batch : dictionary
            A batch of positive and negative associations with the corresponding metabolite and disease indices
        manifold: geoopt.Manifold
            Embedding manifold
        device: string
            The device currently used by pytorch
        metabolite_embedding_module: ManifoldEmbedding
            Pytorch module for embedding metabolites based on indices
        disease_embedding_module: ManifoldEmbedding
            Pytorch module for embedding diseases based on indices

        Returns
        ------
        predicted_associations : 1D tensor on the device
            Predicted associations based on embedding similarities
        true_associations : 1D tensor on the device
            Ground truth binary associations
    """
    true_associations = batch["association"].to(device)
    metabolite_embeddings = metabolite_embedding_module(batch["metabolite"])
    disease_embeddings = disease_embedding_module(batch["disease"])
    dists = manifold.dist(metabolite_embeddings, disease_embeddings)
    predicted_associations = 1/(1+dists) 
    return predicted_associations, true_associations