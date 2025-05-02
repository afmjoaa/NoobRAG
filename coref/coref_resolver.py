# Following line is important even if it's not in use.
from fastcoref import spacy_component
import spacy
from typing import List, Dict, Any, Optional
import logging


class CorefResolver:
    """
    A class for resolving coreferences in text using SpaCy and FastCoref.
    Follows clean architecture principles for maintainability and extensibility.
    """

    def __init__(self,
                 model_name: str = "en_core_web_trf",
                 coref_model_architecture: str = "LingMessCoref",
                 coref_model_path: str = "biu-nlp/lingmess-coref",
                 device: str = "cpu"):
        """
        Initialize the CorefResolver with specified models.

        Args:
            model_name: The SpaCy model to use.
            coref_model_architecture: The architecture for the coref model.
            coref_model_path: The path to the coref model.
            device: The device to use for computation (cpu or cuda).
        """
        self.model_name = model_name
        self.coref_model_architecture = coref_model_architecture
        self.coref_model_path = coref_model_path
        self.device = device
        self.nlp = None

        # Initialize the NLP pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """
        Initialize the SpaCy NLP pipeline with the FastCoref component.
        """
        try:
            self.nlp = spacy.load(self.model_name)

            # Add the FastCoref pipe
            self.nlp.add_pipe("fastcoref", config={
                'model_architecture': self.coref_model_architecture,
                'model_path': self.coref_model_path,
                'device': self.device
            })

            logging.info(f"Successfully initialized NLP pipeline with {self.model_name} and FastCoref")
        except Exception as e:
            logging.error(f"Failed to initialize NLP pipeline: {str(e)}")
            raise

    def resolve_single_text(self, text: str) -> str:
        if not text.strip():
            return text

        try:
            doc = self.nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
            return doc._.resolved_text
        except Exception as e:
            logging.error(f"Error in coreference resolution: {str(e)}")
            return text  # Return original text in case of error

    def resolve_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        resolved_documents = []

        for doc in documents:
            # Create a copy of the document to avoid modifying the original
            resolved_doc = doc.copy()

            # Only process if the 'text' field exists
            if 'text' in resolved_doc:
                resolved_doc['text'] = self.resolve_single_text(resolved_doc['text'])

            resolved_documents.append(resolved_doc)

        return resolved_documents

    def process_batch(self, documents: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
        all_resolved_docs = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            resolved_batch = self.resolve_documents(batch)
            all_resolved_docs.extend(resolved_batch)

        return all_resolved_docs


# Example usage:
if __name__ == "__main__":
    # Example documents
    sample_documents = [
        {
            "doc_id": "<urn:uuid:0cf75b43-d690-4aa6-b3ca-f488ceb28ed9>",
            "text": "Sarah told Emily that she would have to leave if she kept yelling at her in front of the kids.",
            "score": 0.835362315,
            "source": "dense"
        },
        {
            "doc_id": "<urn:uuid:b6541cf6-d442-454a-9e6f-45eaaa237424>",
            "text": "Angela Merkel met Barack Obama yesterday. She greeted him warmly.",
            "score": 0.823291481,
            "source": "dense"
        }
    ]

    # Initialize resolver
    try:
        # First ensure you have the required models installed
        # Uncomment these lines if you need to download the models
        # import os
        # os.system("python -m spacy download en_core_web_trf")
        # os.system("pip install fastcoref")

        # Create resolver
        resolver = CorefResolver()

        # Process documents
        resolved_docs = resolver.resolve_documents(sample_documents)

        print(resolved_docs)

        # Print original vs resolved
        for i, (orig, res) in enumerate(zip(sample_documents, resolved_docs)):
            print(f"Document {i + 1} - Original: {orig['text']}")
            print(f"Document {i + 1} - Resolved: {res['text']}")
            print("-" * 50)

    except Exception as e:
        print(f"Error: {str(e)}")










# from fastcoref import spacy_component
# import spacy
#
# text = """
# Environment in Focus: Green Revolution\nThis content is not assigned to a topic\nTOPIC OF THE WEEK\nENVIRONMENT IN FOCUS:\n“Green Revolution” is a public relations term, probably coined in the 1960s by William Gaud, then Director of the U.S. Agency for international Development, that symbolized the modernization of agriculture in less industrialized countries by means of technological change rather than violent “Red Revolution” (Communism).\nThe essence of the Green Revolution was developing fertilizer-responsive varieties of wheat and rice that would increase national yields of these basic cereals.\nWith the increased national yields, promoters of this agricultural modernization saw the potential for reducing hunger, poverty, misery, and the potential for violent social upheaval that would threaten the geopolitical interests of the United States and other western powers.\nScientific research that underlay the green revolution is generally dated from the establishment of the Mexican Agricultural Program by the Rockefeller Foundation in 1943, and the successful technological changes were well in place by the 1970s in many parts of Asia and Latin America. Green revolution technology is now almost universally the “normal” way to do agriculture in all parts of the world except in parts of Africa.\nFor most of human evolutionary history, people were hunter-gatherers. They ate that which grew with little or no human management. Beginning about 40,000 years ago, and culminating about 10,000 years ago, however, people invented and increasingly used active management or “agriculture” to increase the quantity and security of food supplies, including grains, vegetables, and livestock. With this change, often called the “Neolithic Revolution,” modern humans became more sedentary and lived in villages or cities. A farming class tilled the soil and created the potential for large stores of grain that supported an increasingly thriving urban culture. For most of the past 10,000 years, until about 250 years ago, most (probably over 90 percent) people were tillers of the soil, and the primary human occupation was “farmer.”\nWith the exception of the last 100 years, the primary limitation on agricultural production was the supply of nitrogen to growing plants. The natural nitrogen cycle slowly takes the largely inert gas nitrogen (N2) from the atmosphere and fixes it into soil in forms that can be utilized as nutrients by plants.\nHuman activities, including the production of nitrogen fertilizers, combustion of fossil fuels, and other activities, however, have substantially increased the amounts of nitrogen fixed per year to levels that rival rates of natural nitrogen fixation. Increased levels of nitrogen fixation have already had effects on the atmosphere, terrestrial ecosystems, and aquatic ecosystems. Most of the effects are troublesome, but the still increasing uses of nitrogen fertilizer clearly have made agricultural ecosystems more productive. Nitrogen fertilizer, in a word, was essential to the productivity gains of the green revolution. As human population continues to expand, expansion of the use of nitrogen fertilizer, too, will occur.\nBefore 1900, the only major interventions people could do to increase supplies of nitrogen for crop plants lay in using manure, crop rotations with leguminous crops like clover, and mining of salts like sodium nitrate and natural materials like bird guano. These extra supplies, however, were expensive and limited in size. People were surrounded by billions of tons of inert nitrogen in the air, but it was impossible to bring this material to service of increasing yields in agriculture.\nRELATED NEWS IN FOCUS"
# """
#
# # python -m spacy download en_core_web_trf
# nlp = spacy.load("en_core_web_trf")
# nlp.add_pipe("fastcoref",
#              config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'}
# )
#
# doc = nlp(      # for multiple texts use nlp.pipe
#    text,
#    component_cfg={"fastcoref": {'resolve_text': True}}
# )
#
# print(doc._.resolved_text)


# pip install spacy==3.7.4 fastcoref==2.1.6 transformers==4.34.1 torch