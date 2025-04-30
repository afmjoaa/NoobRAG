from fastcoref import spacy_component
import spacy


# text = 'Sarah told Emily that she would have to leave if she kept yelling at her in front of the kids. Alice goes down the rabbit hole. Where she would discover a new reality beyond her expectations. Angela Merkel met Barack Obama yesterday. She greeted him warmly. Later in the day, they had a long discussion.'
text = """
Environment in Focus: Green Revolution\nThis content is not assigned to a topic\nTOPIC OF THE WEEK\nENVIRONMENT IN FOCUS:\n“Green Revolution” is a public relations term, probably coined in the 1960s by William Gaud, then Director of the U.S. Agency for international Development, that symbolized the modernization of agriculture in less industrialized countries by means of technological change rather than violent “Red Revolution” (Communism).\nThe essence of the Green Revolution was developing fertilizer-responsive varieties of wheat and rice that would increase national yields of these basic cereals.\nWith the increased national yields, promoters of this agricultural modernization saw the potential for reducing hunger, poverty, misery, and the potential for violent social upheaval that would threaten the geopolitical interests of the United States and other western powers.\nScientific research that underlay the green revolution is generally dated from the establishment of the Mexican Agricultural Program by the Rockefeller Foundation in 1943, and the successful technological changes were well in place by the 1970s in many parts of Asia and Latin America. Green revolution technology is now almost universally the “normal” way to do agriculture in all parts of the world except in parts of Africa.\nFor most of human evolutionary history, people were hunter-gatherers. They ate that which grew with little or no human management. Beginning about 40,000 years ago, and culminating about 10,000 years ago, however, people invented and increasingly used active management or “agriculture” to increase the quantity and security of food supplies, including grains, vegetables, and livestock. With this change, often called the “Neolithic Revolution,” modern humans became more sedentary and lived in villages or cities. A farming class tilled the soil and created the potential for large stores of grain that supported an increasingly thriving urban culture. For most of the past 10,000 years, until about 250 years ago, most (probably over 90 percent) people were tillers of the soil, and the primary human occupation was “farmer.”\nWith the exception of the last 100 years, the primary limitation on agricultural production was the supply of nitrogen to growing plants. The natural nitrogen cycle slowly takes the largely inert gas nitrogen (N2) from the atmosphere and fixes it into soil in forms that can be utilized as nutrients by plants.\nHuman activities, including the production of nitrogen fertilizers, combustion of fossil fuels, and other activities, however, have substantially increased the amounts of nitrogen fixed per year to levels that rival rates of natural nitrogen fixation. Increased levels of nitrogen fixation have already had effects on the atmosphere, terrestrial ecosystems, and aquatic ecosystems. Most of the effects are troublesome, but the still increasing uses of nitrogen fertilizer clearly have made agricultural ecosystems more productive. Nitrogen fertilizer, in a word, was essential to the productivity gains of the green revolution. As human population continues to expand, expansion of the use of nitrogen fertilizer, too, will occur.\nBefore 1900, the only major interventions people could do to increase supplies of nitrogen for crop plants lay in using manure, crop rotations with leguminous crops like clover, and mining of salts like sodium nitrate and natural materials like bird guano. These extra supplies, however, were expensive and limited in size. People were surrounded by billions of tons of inert nitrogen in the air, but it was impossible to bring this material to service of increasing yields in agriculture.\nRELATED NEWS IN FOCUS"
"""

# python -m spacy download en_core_web_trf
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("fastcoref",
             config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'}
)

# doc = nlp(text)
# print(doc._.coref_clusters)

# nlp.add_pipe(
#    "fastcoref",
#    config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'}
# )

doc = nlp(      # for multiple texts use nlp.pipe
   text,
   component_cfg={"fastcoref": {'resolve_text': True}}
)

print(doc._.resolved_text)