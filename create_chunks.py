import json
import traceback

outputFile = "caf_chunks.json"
principles = ["Principle A1 - Governance", "Principle A2 - Risk Management", "Principle A3 - Asset Management", "Principle A4 - Supply Chain", "Principle B1 - Service Protection Policies and Processes",
              "Principle B2 - Identity and Access Control", "Principle B3 - Data Security", "Principle B4 - System Security", "Principle B5 - Resilient Networks and Systems", "Principle B6 - Staff Awareness and Training", 
              "Principle C1 - Security Monitoring", "Principle C2 - Proactive Security Event Discovery", "Principle D1 - Response and Recovery Planning", "Principle D2 - Lessons Learned"]

def main():
    dumped_data = []
    temp = {}
    try:
        with open('caf_clean.json', 'r') as f:
            data = json.load(f)
        
        
            #objective_name = list(data.keys())[0]
            print("Keys: ", data.keys())
            #print("Keys2: ", data[objective_name].keys())
            #principle = data[objective_name]['principle']
            for objective_name in data.keys():
                for principle in data[objective_name].keys():
                    for igp in data[objective_name][principle]['IGPs']:
                        #print("Principle: ", principle, "IGP: ", igp)
                        id = igp['id']
                        name = igp['name']
                        text = igp['text']
                        contextualized_to_park = igp['contextualized_to_park']
                        related_mitre = igp['related_mitre']

                        for achieved in igp["status"]["Achieved"]:
                            temp = {
                                "content": achieved,
                                "metadata": {
                                    "objective": objective_name,
                                    "principle": principle,
                                    "id": id,
                                    "name": name,
                                    "text": text,
                                    "status": "Achieved",
                                    "contextualized_to_park": contextualized_to_park,
                                    "related_mitre": related_mitre,
                                }
                            }
                            dumped_data.append(temp)
                        if "Partially Achieved" not in igp["status"]:
                            continue
                        else:
                            for partially_achieved in igp["status"]["Partially Achieved"]:
                                temp = {
                                    "content": partially_achieved,
                                    "metadata": {
                                        "objective": objective_name,
                                        "principle": principle,
                                        "id": id,
                                        "name": name,
                                        "text": text,
                                        "status": "Partially Achieved",
                                        "contextualized_to_park": contextualized_to_park,
                                        "related_mitre": related_mitre,
                                    }
                                }
                                dumped_data.append(temp)

                        for not_achieved in igp["status"]["Not Achieved"]:
                            temp = {
                                "content": not_achieved,
                                "metadata": {
                                    "objective": objective_name,
                                    "principle": principle,
                                    "id": id,
                                    "name": name,
                                    "text": text,
                                    "status": "Not Achieved",
                                    "contextualized_to_park": contextualized_to_park,
                                    "related_mitre": related_mitre,
                                }
                            }
                            dumped_data.append(temp)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        traceback.print_exc()
        return
    
    with open(outputFile, 'w') as outfile:
        json.dump(dumped_data, outfile, indent=2)

main()