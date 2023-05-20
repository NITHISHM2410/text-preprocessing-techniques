# TEXT PREPROCESSING 

  
  
 
# => Text Cleaning Class
One of the primary functions of the text cleansing module is noise reduction. It removes punctuation, special characters, and numbers that can introduce cacophony and hinder sentiment analysis algorithms. In addition, the module handles non-emotional stopwords, such as "and" and "the," which are managed. Eliminating stopwords reduces the dimension of the data, allowing the emphasis to be placed on more significant words and phrases.
Techniques for lemmatization and stemming are also included in the text purification module. These processes reduce words to their fundamental or root forms, thereby combining terms with similar meanings. For example, "run," "running," and "ran" are shortened to "run." This consolidation improves sentiment analysis by capturing the sentiment associated with a word more precisely.By utilising this text cleansing module, sentiment analysis models can benefit from text data that has been refined and standardised. The module's capacity to eliminate noise, manage stopwords, perform lemmatization and stemming, and account for emojis and emoticons greatly improves the accuracy of sentiment analysis and facilitates accurate sentiment classification.


            
           
           
 # => Text Encoding Class
This function encapsulates text for the BERT model and models based on custom vocabulary, as opposed to using the BERT vocabulary. By utilising the softmax layer as the final layer when attempting to predict tokens, we can reduce the computational cost associated with NLP models. When compared to the approximately 30,000 terms contained in the BERT vocabulary, the use of custom vocabulary will result in a reduction in the overall size of the vocabulary. Therefore, this module was implemented to perform both encoding and decoding for bert based on custom vocabulary.         
            

 # => Usage
Download the.py file to your project directory or directly import it to the python notebook in order to utilize either the encoding module or the cleaning module. Each module may be found in the directory that is specific to its purpose. Along with the module, an example Notebook file is also uploaded for users' convenience in the course of implementation.
      
 
