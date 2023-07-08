# TEXT PREPROCESSING 

  
  
 
**Text Cleaning Class**<br>
Text cleaning makes sure that natural language processing systems can read and understand the text they are given. The human speaking system is a mix of random characters, symbols, and punctuation marks. NLP is used to deal with it. Text cleaning is the process of getting rid of these distracting parts. This lets natural language processing models focus on important language information and makes NLP work better.This module cleans up text all at once in a quick and effective way.


<br>
           
           
**Text Encoding Class**<br>
This module includes BERT model usage based on custom vocabulary, as opposed to BERT model usage based on in-built vocabulary. It can be advantageous to employ a custom vocabulary with a smaller size than the BERT model's default vocabulary.A custom vocabulary enables you to reduce the number of tokens in the vocabulary to a subset of domain-specific words that BERT's general text-based vocabulary may not adequately encompass.This change enhances the model's comprehension and effectiveness in a specific domain or application.The original vocabulary of BERT consists of approximately 30,000 elements, which ambiguizes with multiple meanings in the domain. Domain-specific vocabularies enable customized embeddings as we re-train the BERT model to capture domain-specific relationships and nuances, resulting in improved results that can be easily achieved with this module.By using a custom vocabulary, you can reduce the complexity and increases the efficiency of token prediction ability of the softmax layer and make it easier to obtain the score for each token during tasks like MLM or seq2seq tasks. This module provides a decoding capability that decodes an Integer token from a transformer decoder or any seq2seq decoder to String with a single request.Lastly, this module inherits keras layer, allowing us to utilize it as a layer within a keras model.
            

**Usage**<br>
Download the.py file to your project directory or directly import it to the python notebook in order to utilize either the encoding module or the cleaning module. Each module may be found in the directory that is specific to its purpose. Along with the module, an example Notebook file is also uploaded for user's convenience in the course of implementation.
      
 
