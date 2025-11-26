def get_recommendation(level):
    if level == "low":
        return (
            "Low emissions. Continue sustainable habits: eco-friendly travel, low meat use, "
            "low plastic, and energy saving."
        )

    elif level == "medium":
        return (
            "Medium emissions. Reduce meat meals, minimize plastic, use less electricity, "
            "and choose fewer high-impact activities."
        )

    elif level == "high":
        return (
            "High emissions! Switch to EV/public transport, reduce AC use, eat more vegetarian meals, "
            "avoid plastic, and choose eco-certified hotels."
        )

    return "No recommendation available."
