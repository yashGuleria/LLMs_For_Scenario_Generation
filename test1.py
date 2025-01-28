import os
import json
import xml.etree.ElementTree as ET
import xml.dom.minidom
import logging
import random
import tempfile
import shutil
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatCohere
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from typing import List, Dict, Any

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def json_to_xml(json_data):
    """
    Converts JSON data into XML format according to the XDAT schema.

    Args:
        json_data (dict): JSON data to be converted.

    Returns:
        str: A formatted XML string.
    """
    waket = {'A320': 'MEDIUM', 'B738': 'MEDIUM', 'B737': 'MEDIUM', 'B744': 'HEAVY',
             'B734': 'MEDIUM', 'A388': 'HEAVY', 'A333': 'HEAVY'}

    root_ifp = ET.Element("ifp")

    experiment_date = ET.SubElement(root_ifp, "experiment-date")
    ET.SubElement(experiment_date, "day").text = '6'
    ET.SubElement(experiment_date, "month").text = '3'
    ET.SubElement(experiment_date, "year").text = '2022'

    ET.SubElement(root_ifp, "experiment-time").text = '00000'
    ET.SubElement(
        root_ifp, "default-equipment").text = "SSR_MODE_A+SSR_MODE_C+P_RNAV+FMS+FMS_GUIDANCE_VNAV+FMS_GUIDANCE_SPEED"

    for i, item in enumerate(json_data.values()):
        print("ITEM", item)
        initial_flightplans = ET.SubElement(
            root_ifp, "initial-flightplans", key=f"initial-flightplans:{i}")
        ET.SubElement(initial_flightplans, "usage").text = "ALL"
        ET.SubElement(initial_flightplans, "time").text = str(item["time"])
        ET.SubElement(initial_flightplans, "callsign").text = f"SQ1{i}"
        ET.SubElement(initial_flightplans, "type").text = item["type"]
        ET.SubElement(initial_flightplans,
                      "waketurb").text = waket[item["type"]]

        dep = ET.SubElement(initial_flightplans, "dep")
        ET.SubElement(dep, "af").text = item["departure"]["af"]

        des = ET.SubElement(initial_flightplans, "des")
        ET.SubElement(des, "af").text = item["destination"]["af"]

        for route in item["air_route"]:
            ET.SubElement(initial_flightplans, "air_route").text = route

        ET.SubElement(initial_flightplans, "rfl").text = item["rfl"]

        init = ET.SubElement(initial_flightplans, "init")
        pos = ET.SubElement(init, "pos")
        ET.SubElement(pos, "lat").text = item["initial_position"]["latitude"]
        ET.SubElement(pos, "lon").text = item["initial_position"]["longitude"]

        alt = ET.SubElement(
            init, "alt", units=item["initial_position"]["altitude"][:2])
        alt.text = item["initial_position"]["altitude"][2:]
        ET.SubElement(init, "hdg").text = str(
            item["initial_position"]["heading"])

    xdat_content = ET.tostring(
        root_ifp, encoding='UTF-8', xml_declaration=True)
    dom = xml.dom.minidom.parseString(xdat_content)
    return dom.toprettyxml()


output_dir = "prompt_output"
os.makedirs(output_dir, exist_ok=True)
file_path = "Data_bundlefinal.pdf"


def create_vector_db(file_path) -> Chroma:
    logger.info(f"Creating vector DB from file: {file_path}")
    loader = UnstructuredPDFLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["-----------", "\n\n", "\n"], chunk_size=550, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large", show_progress=True)
    vector_db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name="myRAG"
    )
    logger.info("Vector DB created")

    return vector_db


def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(
        f"Processing question: {question} using model: {selected_model}")
    llm = ChatCohere(cohere_api_key='API_KEY',
                     model="command-r", temperature=0, streaming=True,
                     preamble="""You are an AI system that generates traffic scenarios for a simulator.
                    - When the user requests for a scenario, strictly give the output in JSON based on schema_1, otherwise reply normally.
                    - Do not give python code.
                    - Refer to the chat_history to understand the context of the conversation before replying.
                    - Use complete air_routes from the Data Bundle when generating outputs.
                    - Ensure that the aircraft are separated by a minimum time 100 seconds.
                    
                    ----------------------------------------
                        schema_1:
                        {
                        "aircraft 1": {
                        "departure": {
                        "af": "ICAO code of departure airfield here"
                        },
                        "initial_position": {
                        "latitude": "latitude in DMS format for example 023106.70N",
                        "longitude": "latitude in DMS format for example 1035709.81E",
                        "altitude": " initial flight level / initial Altitude reading, for example FL300",
                        "heading": "heading in degrees for example 32.05"
                        },
                        "air_route": "list of waypoints that make up the air route, for example [\"RAXIM\", \"OTLON\", \"VISAT\", \"DUBSA\", \"DAMOG\", \"DOLOX\", \"DUDIS\"]",
                        "destination": {
                        "af": "ICAO code of destination airfield here"
                        },
                        "time": "starting time of aircraft initialization in the simulator as an integer representing seconds, for example 300",
                        "type": "type of aircraft, for example A320",
                        "rfl": "Final flight level of the aircraft",},
                        "aircraft 2": {},
                        "aircraft n": {},
                        }
                        where n is the number of aircraft in the scenario.

                    ---------------------------------------------
                    Only use the Available Aircraft Types:
                    • A320
                    • B737
                    • B738
                    • B734
                    • B744
                    • A388
                    • A333
                    
                    ---------------------------------------------
                   If the number of aircraft are not specified, use the following range:
                   • Type 1 traffic density: 1 to 4 aircraft.
                   • Type 2 traffic density: 5 to 9 aircraft.
                   • Type 3 traffic density: 10 to 20 aircraft.
                   Otherwise, ensure to generate the same number of aircraft that are specified.""")

    retriever = vector_db.as_retriever(
        search_type="mmr", search_kwargs={"fetch_k": 28, "k": 16, "lambda_mult": 0.5})

    extracted_docs = retriever.invoke(question)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    formatted_docs = format_docs(extracted_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", """User prompt: {question}
             --------------------------------------------
             The Data bundle contains examples of airways, along with their corresponding list of waypoints that define the air routes for each airway. When creating scenarios, refer to these examples and ensure to match the complete list of waypoints provided for each respective airway.
             
             Data bundle: {context}""")
        ]
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(
        {"context": formatted_docs,  "question": question})

    logger.info("Question processed and response generated")
    return response


def str2dict(response):
    """
    Extract JSON content from a response string enclosed in triple backticks.
    """
    import re
    pattern = r"```json\n(.*?)\n```"
    matches = re.findall(pattern, response, re.DOTALL)
    json_d = {}
    if matches:
        for item in matches:
            json_d.update(json.loads(item))
    else:
        json_d = {}
    return json_d


for i in range(1):
    try:
        # Initialize the vector database
        vector_db = create_vector_db(file_path)

        # Define the retriever
        retriever = vector_db.as_retriever(
            search_type="mmr", search_kwargs={"fetch_k": 28, "k": 16, "lambda_mult": 0.5}
        )

        # Extract relevant context documents
        question = "generate 4 aircraft in sector 6"
        extracted_docs = retriever.invoke(question)

        # Format the documents into a readable context string
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        formatted_docs = format_docs(extracted_docs)

        # Prepare the LLM chain with a prompt
        llm = ChatCohere(
            cohere_api_key='njPNUMUWPRMIoHRPoV8xSJvucU2sEZw99puyga7r',
            model="command-r",
            temperature=0,
            streaming=True,
            preamble=""""You are an AI system that generates traffic scenarios for a simulator.
                    - When the user requests for a scenario, strictly give the output in JSON based on schema_1, otherwise reply normally.
                    - Do not give python code.
                    - Refer to the chat_history to understand the context of the conversation before replying.
                    - Use complete air_routes from the Data Bundle when generating outputs.
                    - Ensure that the aircraft are separated by a minimum time 100 seconds.
                    
                    ----------------------------------------
                        schema_1:
                        {
                        "aircraft 1": {
                        "departure": {
                        "af": "ICAO code of departure airfield here"
                        },
                        "initial_position": {
                        "latitude": "latitude in DMS format for example 023106.70N",
                        "longitude": "latitude in DMS format for example 1035709.81E",
                        "altitude": " initial flight level / initial Altitude reading, for example FL300",
                        "heading": "heading in degrees for example 32.05"
                        },
                        "air_route": "list of waypoints that make up the air route, for example [\"RAXIM\", \"OTLON\", \"VISAT\", \"DUBSA\", \"DAMOG\", \"DOLOX\", \"DUDIS\"]",
                        "destination": {
                        "af": "ICAO code of destination airfield here"
                        },
                        "time": "starting time of aircraft initialization in the simulator as an integer representing seconds, for example 300",
                        "type": "type of aircraft, for example A320",
                        "rfl": "Final flight level of the aircraft",},
                        "aircraft 2": {},
                        "aircraft n": {},
                        }
                        where n is the number of aircraft in the scenario.

                    ---------------------------------------------
                    Only use the Available Aircraft Types:
                    • A320
                    • B737
                    • B738
                    • B734
                    • B744
                    • A388
                    • A333
                    
                    ---------------------------------------------
                   If the number of aircraft are not specified, use the following range:
                   • Type 1 traffic density: 1 to 4 aircraft.
                   • Type 2 traffic density: 5 to 9 aircraft.
                   • Type 3 traffic density: 10 to 20 aircraft.
                   Otherwise, ensure to generate the same number of aircraft that are specified."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("human", """User prompt: {question}
                 --------------------------------------------
                 
                 The Data bundle contains examples of airways, along with their corresponding list of waypoints that define the air routes for each airway. When creating scenarios, refer to these examples and ensure to match the complete list of waypoints provided for each respective airway.
                 
                 Data bundle: {context}""")
            ]
        )

        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        # Invoke the chain to get the response
        response = chain.invoke(
            {"context": formatted_docs, "question": question})
        print("RESPONSE", response)

        if hasattr(response, 'token_count'):
            logger.info(f"Token count: {response.token_count}")
        else:
            logger.warning(
                "Response object has no 'token_count' attribute. Skipping token count logging.")

        # Extract JSON content from the response
        json_d = str2dict(response)

        if json_d:
            # Convert JSON to XDAT XML format
            xdat_content = json_to_xml(json_d)

            # Save the XML content to a file
            file_path = os.path.join(output_dir, f"scenario_{i+1}.xdat")
            with open(file_path, "w") as file:
                file.write(xdat_content)

            logger.info(f"Iteration {i+1}: Output saved to {file_path}")
        else:
            logger.debug(f"Raw response: {response}")
            logger.warning(f"Iteration {i+1}: No valid JSON response")

        # Clean up vector database
        if vector_db:
            vector_db.delete_collection()
            logger.info(f"Vector DB for iteration {i+1} deleted.")

    except Exception as e:
        logger.error(f"Iteration {i+1} failed with error: {e}")

logger.info("Processing complete.")
