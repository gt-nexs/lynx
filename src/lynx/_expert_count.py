num_selected_experts_policy = 0
num_selected_experts_base = 0

def add_num_selected_experts_policy(value):
    global num_selected_experts_policy
    num_selected_experts_policy += value

def add_num_selected_experts_base(value):
    global num_selected_experts_base
    num_selected_experts_base += value

def get_num_selected_experts_policy():
    return num_selected_experts_policy

def get_num_selected_experts_base():
    return num_selected_experts_base
    