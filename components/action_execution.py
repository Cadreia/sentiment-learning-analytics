# components/action_execution.py
def action_execution(final_actions):
    executed_actions = []
    for action in final_actions:
        if "Flag Student" in action or "Suggest Intervention" in action:
            student_id = action.split("Student ")[1].split(" ")[0]
            message = (f"Send Automated Message to Student {student_id}: 'We noticed you're at risk. Let's schedule a "
                       f"meeting.'")
            executed_actions.append(message)

            if "Dropout Risk" in action and float(action.split("Dropout Risk: ")[1].split(")")[0]) > 0.7:
                intervention = f"Trigger Intervention for Student {student_id}: Mandatory counseling session"
                executed_actions.append(intervention)

        if "Adapt Teaching Strategy" in action:
            student_id = len(executed_actions)
            content = f"Adjust Content Delivery for Student {student_id}: "
            if "interactive content" in action:
                content += "Add more interactive materials"
            elif "practice problems" in action:
                content += "Provide additional practice problems"
            else:
                content += "Focus on conceptual videos"
            executed_actions.append(content)

        executed_actions.append(action)

    return executed_actions
