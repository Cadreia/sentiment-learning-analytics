# def action_execution(final_actions):
#     executed_actions = []
#     for action in final_actions:
#         if "Flag Student" in action or "Suggest Intervention" in action:
#             student_id = action.split("Student ")[1].split(" ")[0]
#             message = (f"Send Automated Message to Student {student_id}: 'We noticed you're at risk. Let's schedule a "
#                        f"meeting.'")
#             executed_actions.append(message)
#
#             if "Dropout Risk" in action and float(action.split("Dropout Risk: ")[1].split(")")[0]) > 0.7:
#                 intervention = f"Trigger Intervention for Student {student_id}: Mandatory counseling session"
#                 executed_actions.append(intervention)
#
#         if "Adapt Teaching Strategy" in action:
#             student_id = len(executed_actions)
#             content = f"Adjust Content Delivery for Student {student_id}: "
#             if "interactive content" in action:
#                 content += "Add more interactive materials"
#             elif "practice problems" in action:
#                 content += "Provide additional practice problems"
#             else:
#                 content += "Focus on conceptual videos"
#             executed_actions.append(content)
#
#         executed_actions.append(action)
#
#     return executed_actions


import pandas as pd


def action_execution(fused_decisions, student_id):
    """
    Execute actions based on fused decisions.

    Parameters:
    fused_decisions (dict): Fused decisions from fusion and adaptation.
    student_id (str): Student ID.

    Returns:
    dict: Executed actions.
    """
    # Extract decisions
    student_id = fused_decisions["student_id"]
    feedback = fused_decisions["feedback"]
    teaching_strategy = fused_decisions["teaching_strategy"]
    interventions = fused_decisions["interventions"]
    priority = fused_decisions["fused_strategy"]["priority"]
    review_flag = fused_decisions["review_flag"]

    # Send automated messages
    messages = send_automated_messages(feedback, interventions, student_id, priority)

    # Adjust content delivery
    content_adjustments = adjust_content_delivery(teaching_strategy, priority)

    # Trigger interventions
    triggered_interventions = trigger_interventions(interventions, student_id)

    # Compile executed actions
    executed_actions = {
        "student_id": student_id,
        "messages": messages,
        "content_adjustments": content_adjustments,
        "triggered_interventions": triggered_interventions,
        "review_flag": review_flag,
        "priority": priority
    }

    return executed_actions


def send_automated_messages(feedback, interventions, student_id, priority):
    """
    Simulate sending automated messages to the student or instructor.

    Parameters:
    feedback (str): Feedback message.
    interventions (list): List of interventions.
    student_id (str): Student ID.
    priority (str): Priority level ("Low", "Medium", "High").

    Returns:
    list: List of messages sent.
    """
    messages = []
    messages.append(f"To Student {student_id}: {feedback}")

    if priority in ["Medium", "High"]:
        messages.append(f"To Instructor: Please review Student {student_id}. Priority: {priority}.")
        messages.append(f"Interventions for Student {student_id}: {', '.join(interventions)}")

    return messages


def adjust_content_delivery(teaching_strategy, priority):
    """
    Simulate adjusting content delivery based on teaching strategy.

    Parameters:
    teaching_strategy (str): Recommended teaching strategy.
    priority (str): Priority level.

    Returns:
    str: Content adjustment description.
    """
    return f"Content adjusted for priority {priority}: {teaching_strategy}"


def trigger_interventions(interventions, student_id):
    """
    Simulate triggering interventions for the student.

    Parameters:
    interventions (list): List of interventions.
    student_id (str): Student ID.

    Returns:
    list: List of triggered interventions.
    """
    return [f"Triggered for Student {student_id}: {intervention}" for intervention in interventions]
