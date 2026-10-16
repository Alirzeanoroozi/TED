### errors
## 1. number of domains is not correct
## 2. not feasible boundaries (not ordered or overlapping or out of range)


def split_domain(domain):
    return domain.split('_')

def parse_domains(predicted, actual):
    if predicted.count('*') != actual.count('*'):
        raise ValueError(f"number of domains is not correct {predicted.count('*')} in predicted and {actual.count('*')} in actual")
    
    parsed_actual_bounds = []
    parsed_actual_cath = []
    parsed_predicted_bounds = []
    parsed_predicted_cath = []
    
    predicted_domains = predicted.split('*')
    actual_domains = actual.split('*')
    for i in range(len(predicted_domains)):
        parsed_predicted_bounds.append(split_domain(predicted_domains[i].strip().split('|')[0].strip()))
        parsed_predicted_cath.append(predicted_domains[i].strip().split('|')[1].strip())
        parsed_actual_bounds.append(split_domain(actual_domains[i].strip().split('|')[0].strip()))
        parsed_actual_cath.append(actual_domains[i].strip().split('|')[1].strip())
    return parsed_predicted_bounds, parsed_predicted_cath, parsed_actual_bounds, parsed_actual_cath

if __name__ == "__main__":
    parsed_predicted_bounds, parsed_predicted_cath, parsed_actual_bounds, parsed_actual_cath = parse_domains("10-177 | 3.90.1150.10 * 187-375 | 3.90.50.10" ,"12-158_312-364 | 3.40.50.12650 * 164-306 | 3.60.15.10")
    print(parsed_predicted_bounds, parsed_predicted_cath, parsed_actual_bounds, parsed_actual_cath)