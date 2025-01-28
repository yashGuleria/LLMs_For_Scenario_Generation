import time
from nltk.tokenize import sent_tokenize
import logging
import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, AIMessage
import json
import re
import xml.etree.ElementTree as ET
import xml.dom.minidom
import random
from datetime import datetime

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_vector_db(file_path):
    """Create a vector database from a PDF file."""
    logger.info(f"Creating vector DB from file: {file_path}")
    loader = UnstructuredPDFLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["-----------", "\n\n", "\n"],
        chunk_size=550,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large", show_progress=True)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="myRAG"
    )
    logger.info("Vector DB created")
    return vector_db


def get_history(chat_messages):
    """Convert chat messages to the format expected by the model."""
    chat_history = []
    for m in chat_messages:
        if m['role'] == 'user':
            chat_history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            chat_history.append(AIMessage(content=m["content"]))
    return chat_history


def process_question(question, vector_db, chat_messages):
    """Process a user question using the vector database and language model."""
    logger.info(f"Processing question: {question}")

    llm = ChatCohere(
        # cohere_api_key='njPNUMUWPRMIoHRPoV8xSJvucU2sEZw99puyga7r',
        # cohere_api_key='PizzEEKA89RGBhRMfywZXTFNxA3Zk5svE8WDaHK1',
        cohere_api_key="SKnc6YwodGhQ9lf0n44w7ymcruesLnDM4yWid7nP",
        # cohere_api_key="3fJGM2WgyYQuHV1H0m82OFgcdaPVWlebrrRNahpg",
        model="command-r",
        temperature=0.5,
        streaming=True,
        preamble="""
                   You are an AI system that generates traffic scenarios for an air traffic simulator.

                    Core Requirements:
                    1. Output Format:
                    - Provide output in JSON format following schema_1 when scenarios are requested
                    - Otherwise, respond conversationally
                    - Never provide Python code in responses
                    - STRICTLY ADHERE TO PROMPT REQUIREMENTS OF AIRWAYS AND AIRCRAFT TYPES.

                    2. Scenario Generation Rules:
                    - When same airway is requested, use a single airway.
                    - Use COMPLETE AIR ROUTES for the airways from the Data Bundle
                    - STRICTLY ENSURE THAT  DEPARTURE AND DESTINATION AIRPORTS MATCH THE AIRWAY.
                    - ONLY USE AIRWAYS AND DEPARTURE AND DESTINATION AIRFIELDS FROM THE DATA BUNDLE
                    - STRICTLY ENFORCE requested time separation between aircraft
                    - STRICTLY ENFORCE airway requirements,  example: same airways.
                    - If prompt asks for same airways, ensure that all aircraft are on the same airway.
                    - Do not interpret M771 and M771R as same airways. THEY ARE DIFFERENT

                    3. Aircraft Type Distribution:
                    - STRICTLY ENFORCE requested aircraft types.
                    - Available Types:
                    * A320, A333, A388, B734, B737, B738, B744

                    5. Default Parameters (when not specified):
                    - Aircraft count ranges:
                    * Type 1: 1-4 aircraft
                    * Type 2: 5-9 aircraft
                    * Type 3: 10-20 aircraft
                    - Default time separation: 100 seconds

                    Schema Definition-  schema_1:
                    {
                        "aircraft 1": {
                            "departure": {
                                "af": "ICAO code of departure airfield"
                            },
                            "initial_position": {
                                "latitude": "latitude in DMS format (e.g., 023106.70N)",
                                "longitude": "longitude in DMS format (e.g., 1035709.81E)",
                                "altitude": "initial flight level/altitude (e.g., FL300)",
                                "heading": "heading in degrees (e.g., 32.05)"
                            },
                            "air_route": "waypoint list (e.g., ["RAXIM", "OTLON", "VISAT", "DUBSA", "DAMOG", "DOLOX", "DUDIS"])",
                            "destination": {
                                "af": "ICAO code of destination airfield"
                            },
                            "time": "initialization time in seconds (integer, e.g., 300)",
                            "type": "aircraft type from available list",
                            "rfl": "final flight level"
                        },
                        "aircraft 2": {},
                        "aircraft n": {}
                    }
                    """
    )

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"fetch_k": 19, "k": 16, "lambda_mult": 0.5}
    )

    extracted_docs = retriever.invoke(question)
    formatted_docs = "\n\n".join(doc.page_content for doc in extracted_docs)

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", """User prompt: {question}
                    --------------------------------------------
                    The Data bundle contains examples of airways, along with their corresponding list of waypoints that define the air routes for each airway. When creating scenarios, refer to these examples and ensure to match the complete list of waypoints provided for each respective airway. IF PROMPT REQUIRES SAME AIRWAY, USE AIRWAY N892 FOR ALL AIRCRAFT. Do not swap destinations and departures airfields for an airway.

                    Data bundle: {context}
                   """),
    ])

    chat_history = get_history(chat_messages)

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "context": formatted_docs,
        "chat_history": chat_history,
        "question": question
    })

    return response


def str2dict(response):
    """Process the response to extract JSON."""
    pattern = r"```json\n(.*?)\n```"
    matches = re.findall(pattern, response, re.DOTALL)
    json_d = {}
    if matches:
        for item in matches:
            # Parse the JSON string
            parsed_json = json.loads(item)
            # If it's a list with a single dictionary, take that dictionary
            if isinstance(parsed_json, list) and len(parsed_json) > 0:
                parsed_json = parsed_json[0]
            # Update our result dictionary
            if isinstance(parsed_json, dict):
                json_d.update(parsed_json)
    else:
        json_d = ''
    return json_d


def json_to_xml(json_data):
    """Convert JSON data to XML format."""
    # waket = {'A320': 'MEDIUM', 'B738': 'MEDIUM', 'B737': 'MEDIUM', 'B744': 'HEAVY',
    #          'B734': 'MEDIUM', 'A388': 'HEAVY', 'A333': 'HEAVY'}

    waket = {
        'A320': 'MEDIUM', 'Airbus A320': 'MEDIUM',
        'B738': 'MEDIUM', 'Boeing 738': 'MEDIUM',
        'B737': 'MEDIUM', 'Boeing 737': 'MEDIUM',
        'B744': 'HEAVY', 'Boeing 744': 'HEAVY',
        'B734': 'MEDIUM', 'Boeing 734': 'MEDIUM',
        'A388': 'HEAVY', 'Airbus A388': 'HEAVY',
        'A333': 'HEAVY', 'Airbus A333': 'HEAVY'
    }

    root_ifp = ET.Element("ifp")

    experiment_date = ET.SubElement(root_ifp, "experiment-date")
    day = ET.SubElement(experiment_date, "day")
    day.text = '6'
    month = ET.SubElement(experiment_date, "month")
    month.text = '3'
    year = ET.SubElement(experiment_date, "year")
    year.text = '2022'

    experiment_time = ET.SubElement(root_ifp, "experiment-time")
    experiment_time.text = '00000'

    default_equipment = ET.SubElement(root_ifp, "default-equipment")
    default_equipment.text = "SSR_MODE_A+SSR_MODE_C+P_RNAV+FMS+FMS_GUIDANCE_VNAV+FMS_GUIDANCE_SPEED"

    for aircraft_key, item in json_data.items():
        if not isinstance(item, dict):
            continue

        initial_flightplans = ET.SubElement(
            root_ifp, "initial-flightplans", key=f"initial-flightplans:{aircraft_key.split()[-1]}")

        usage = ET.SubElement(initial_flightplans, "usage")
        usage.text = "ALL"
        time = ET.SubElement(initial_flightplans, "time")
        time.text = str(item["time"])

        callsign = ET.SubElement(initial_flightplans, "callsign")
        callsign.text = f"SQ{aircraft_key.split()[-1]}"
        rules = ET.SubElement(initial_flightplans, "rules")
        squawk = ET.SubElement(initial_flightplans, "squawk", units="octal")
        squawk.text = ''.join(str(random.randint(0, 7)) for _ in range(4))

        aircraft_type = ET.SubElement(initial_flightplans, "type")
        aircraft_type.text = item["type"]
        waketurb = ET.SubElement(initial_flightplans, "waketurb")
        waketurb.text = waket[item["type"]]
        equip = ET.SubElement(initial_flightplans, "equip")
        vehicle_type = ET.SubElement(initial_flightplans, "vehicle_type")

        dep = ET.SubElement(initial_flightplans, "dep")
        dep_af = ET.SubElement(dep, "af")
        dep_af.text = item["departure"]["af"]
        dep_rwy = ET.SubElement(dep, "rwy")

        des = ET.SubElement(initial_flightplans, "des")
        des_af = ET.SubElement(des, "af")
        des_af.text = item["destination"]["af"]

        for route in item["air_route"]:
            air_route = ET.SubElement(initial_flightplans, "air_route")
            air_route.text = route

        rfl = ET.SubElement(initial_flightplans, "rfl")
        rfl.text = item["rfl"][2:]

        init = ET.SubElement(initial_flightplans, "init")
        pos = ET.SubElement(init, "pos")
        lat = ET.SubElement(pos, "lat")
        lat.text = item["initial_position"]["latitude"]
        lon = ET.SubElement(pos, "lon")
        lon.text = item["initial_position"]["longitude"]

        freq = ET.SubElement(init, "freq")
        freq.text = 'SINRADS1'
        alt = ET.SubElement(
            init, "alt", units=item["initial_position"]["altitude"][:2])
        alt.text = item["initial_position"]["altitude"][2:]
        hdg = ET.SubElement(init, "hdg")
        hdg.text = str(item["initial_position"]["heading"])

    xdat_content = ET.tostring(
        root_ifp, encoding='UTF-8', xml_declaration=True)
    dom = xml.dom.minidom.parseString(xdat_content)
    pretty_xdat = dom.toprettyxml()

    return pretty_xdat


def save_xml(xml_content, iteration, output_dir="generated_scenarios_prompt3_7jan_temp0.5"):
    """Save XML content to a file."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename
    filename = f"scenario_{timestamp}_iteration_{iteration}.xdat"
    filepath = os.path.join(output_dir, filename)

    # Save the file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(xml_content)

    logger.info(f"Saved XML file: {filepath}")
    return filepath


def main():
    # test promptlevel3_1destinations."
    # prompt = "Generate a 12 aircraft scenario in sector 6. All aircraft should be A388. Separation time between aircraft must be 150 seconds. The first aircraft should start at time 0. Ensure to use correct airways."
    # prompt = "Generate 4 aircraft on airway M758."
    # prompt = "Generate a Type 1 scenario in sector 6 where all aircraft are on same airway."
    prompt = "Generate 3 aircraft on airway M758, airway N884 and airway M761. The aircraft should be separated by 200 seconds with the first aircraft starting at 100 seconds."
    # prompt = "Generate 12 aircraft in sector 6."
    pdf_path = "Data_bundlefinal2_6jan2025.pdf"

    # Process the prompt
    num_iterations = 50
    for i in range(num_iterations):
        logger.info(f"\nStarting Iteration {i+1}")

        # create fresh vector DB
        vector_db = create_vector_db(pdf_path)

        # initialize fresh chat messages for each iteration
        messages = []

        try:
            # get response
            response = process_question(prompt, vector_db, messages)
            print(prompt)
            json_output = str2dict(response)

            logger.info(f"Raw response from iteration {i+1}: ")
            logger.info(response)

            # Generate and save xml if there is a valid json
            if json_output:
                try:
                    xml_content = json_to_xml(json_output)
                    saved_file = save_xml(xml_content, i+1)
                    logger.info(f"saved XML fle: {saved_file}")
                except Exception as e:
                    logger.error(
                        f" error generating/saving XML for iteration {i+1} : {str(e)}")
            else:
                logger.warning(
                    f" No valid JSON to convert to XML in iteration {i+1}")

        except Exception as e:
            logger.error(f"Error in Iteration {i+1} : {str(e)}")

        finally:
            # clean the vector_db after each iteration
            try:
                vector_db.delete_collection()
            except Exception as e:
                logger.error(
                    f" Error deleting VECTOR_DB for iteration {i+1}: {str(e)}")

        # Add 10 second delay before next iteration (except for the last iteration)
        if i < num_iterations - 1:
            logger.info("Waiting 10 seconds before next iteration...")
            time.sleep(10)


if __name__ == "__main__":
    main()
