from __future__ import annotations

import collections
import itertools
import math
from pathlib import Path
from typing import Dict, List

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, dash_table, html
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).parent

PRICE_DATA = pd.read_excel(BASE_DIR / "dashboard_food_recipe.xlsx")
RECIPE_INFO = pd.read_excel(BASE_DIR / "recipes_info.xlsx")

# Dietary preference options
DIET_OPTIONS = ["Vegan", "Halal", "Low-carb"]

NAME_MAP = {
    "Egg_Fried_Noodles": "Egg Fried Noodles",
    "Chicken_Fried_Rice": "Chicken Fried Rice",
    "Chicken_Curry": "Chicken Curry",
    "Tomato_&_Egg_Stir-fry": "Tomato & Egg Stir-fry",
    "Pasta_Bolognese": "Pasta Bolognese",
    "Beef_Lasagne": "Beef Lasagne",
    "Vegetable_Soup": "Vegetable Soup",
    "Broccoli_Cheese_Bake": "Broccoli Cheese Bake",
}

INFO_TO_PRICE = {v: k for k, v in NAME_MAP.items()}

INGREDIENT_SETS = (
    PRICE_DATA.groupby("Recipe_Name")["Ingredient_Name"]
    .apply(lambda series: sorted(set(series)))
    .to_dict()
)

NUTRITION_SCORE = (
    PRICE_DATA.groupby("Recipe_Name")["Nutrition_Score_0_100"].mean().to_dict()
)

INGREDIENT_WEIGHT = (
    PRICE_DATA.groupby(["Recipe_Name", "Ingredient_Name"])["Ingredient_Weight_g"].sum()
)




INGREDIENT_AVG_PRICE = (
    PRICE_DATA.groupby("Ingredient_Name")["Price_AUD_per_unit"].mean().sort_values()
)

INGREDIENT_OPTIONS = [
    {"label": name.replace("_", " "), "value": name}
    for name in sorted(PRICE_DATA["Ingredient_Name"].unique())
]

RECIPE_IMAGE_MAP = {
    "Egg_Fried_Noodles": "dish/egg-fried-noodles.jpg",
    "Chicken_Fried_Rice": "dish/Chicken-Fried-Rice.jpg",
    "Chicken_Curry": "dish/Chicken-Curry.jpg",
    "Tomato_&_Egg_Stir-fry": "dish/stir-fried-tomato-and-egg.jpg",
    "Pasta_Bolognese": "dish/Bolognese.jpg",
    "Beef_Lasagne": "dish/Beef_Lasagne.jpg",
    "Vegetable_Soup": "dish/soup.jpg",
    "Broccoli_Cheese_Bake": "dish/broccoli_cheese.jpg",
}

INGREDIENT_IMAGE_FILES = {
    "Tomato": "ingredient/tomato.jpg",
    "Egg": "ingredient/egg.jpg",
    "Broccoli": "ingredient/Broccoli.jpg",
    "Chicken_breast": "ingredient/Chicken_breast.jpg",
    "Rice": "ingredient/rice.jpg",
    "Beef_mince": "ingredient/beef_mince.jpg",
    "Cheese": "ingredient/cheese.jpg",
    "Pasta": "ingredient/Pasta.jpg",
    "Carrot": "ingredient/Carrot.jpg",
    "Onion": "ingredient/onions.jpg",
}


def build_ingredient_table() -> List[Dict[str, str]]:
    rows = []
    for ingredient, price in INGREDIENT_AVG_PRICE.items():
        label = ingredient.replace("_", " ")
        image_path = INGREDIENT_IMAGE_FILES.get(ingredient)
        image_md = f"![{label}](/assets/{image_path})" if image_path else label
        rows.append(
            {
                "Image": image_md,
                "Ingredient": label,
                "Avg price": f"${price:.2f}",
            }
        )
    return rows


INGREDIENT_TABLE_DATA = build_ingredient_table()

taste_tokens = set()
for taste_value in RECIPE_INFO["Taste"].dropna():
    for token in str(taste_value).split(";"):
        cleaned = token.strip()
        if cleaned:
            taste_tokens.add(cleaned.title())
TASTE_OPTIONS = ["All tastes"] + sorted(taste_tokens)
FAVORITE_OPTIONS = sorted(RECIPE_INFO["Recipe_Name"].unique())
DEFAULT_SERVINGS = 2


def traffic_light(score: float) -> str:
    if math.isnan(score):
        return "No data"
    if score >= 80:
        return "Green"
    if score >= 60:
        return "Amber"
    return "Red"


def dietary_check(diet_preference: str, recipe_diet: str) -> bool:
    """
    Check if a recipe matches the selected dietary preference.

    Args:
        diet_preference: The dietary preference selected by user ("Vegan", "Halal", "Low-carb")
        recipe_diet: The diet category from the recipe's 'Diet' column

    Returns:
        bool: True if the recipe matches the dietary preference, False otherwise
    """
    if not diet_preference:
        return True

    # Convert recipe diet string to lowercase for case-insensitive matching
    recipe_diet_lower = str(recipe_diet).lower()

    # Check for specific dietary preferences
    if diet_preference == "Vegan":
        # Vegan matches vegetarian recipes (since all vegetarian recipes are vegan-suitable)
        return "vegetarian" in recipe_diet_lower
    elif diet_preference == "Halal":
        # Check if recipe contains "halal" in its diet description
        return "halal" in recipe_diet_lower
    elif diet_preference == "Low-carb":
        # Check if recipe contains "low-carb", "low carb", or similar terms
        return any(term in recipe_diet_lower for term in ["low-carb", "low carb", "keto", "low carbohydrate"])

    return True





def plan_recipes(
    budget: float,
    preferences: List[str],
    min_total: float | None = None,
    taste_preference: List[str] | str | None = None,
    servings: int = 1,
    favorites: List[str] | None = None,
    required_ingredients: List[str] | None = None,
) -> pd.DataFrame:
    records = []
    # Handle taste_preference as either a list or string
    if isinstance(taste_preference, list):
        taste_target = taste_preference[0] if taste_preference else ""
    else:
        taste_target = (taste_preference or "").strip()
    taste_target = taste_target.title()
    if taste_target == "All Tastes":
        taste_target = ""
    favorites = favorites or []
    for _, row in RECIPE_INFO.iterrows():
        recipe_name = row["Recipe_Name"]
        price_name = INFO_TO_PRICE.get(recipe_name)
        if not price_name:
            continue
        ingredients = INGREDIENT_SETS.get(price_name, [])
        # Filter by dietary preferences if specified
        recipe_diet = row.get("Diet", "")
        if not all(dietary_check(pref, recipe_diet) for pref in preferences):
            continue
        # Filter by required ingredients if specified
        if required_ingredients:
            # Check if all required ingredients are present in the recipe
            if not all(ing in ingredients for ing in required_ingredients):
                continue
        taste_value = str(row.get("Taste", "")).strip()
        if taste_target:
            if not taste_value:
                continue
            tokens = {
                token.strip().title()
                for token in taste_value.split(";")
                if token.strip()
            }
            if taste_target not in tokens:
                continue
        cost = row.get("Cost_per_person_AUD")
        if math.isnan(cost):
            continue
        records.append(
            {
                "Recipe_Name": recipe_name,
                "Price_Name": price_name,
                "Description": row.get("Description", ""),
                "Cuisine": row.get("Cuisine", ""),
                "Taste": row.get("Taste", ""),
                "Recipe_Link": row.get("Recipe_Link", ""),
                "Cost_per_person_AUD": cost,
                "Nutrition_Score": NUTRITION_SCORE.get(price_name, math.nan),
                "Ingredients": ingredients,
            }
        )
    candidate_df = pd.DataFrame(records)
    if candidate_df.empty:
        return candidate_df
    candidate_df = candidate_df.sort_values("Cost_per_person_AUD").reset_index(drop=True)
    records_list = candidate_df.to_dict("records")
    n = len(records_list)
    best_subset = None
    best_total = -math.inf
    min_total = 0 if min_total is None else min_total
    servings = max(servings, 1)

    favorite_indices = {
        idx for idx, rec in enumerate(records_list) if rec["Recipe_Name"] in favorites
    }

    def search(require_favorites: bool) -> tuple[tuple[int, ...] | None, float]:
        best_combo = None
        best_value = -math.inf
        for r in range(1, n + 1):
            for combo in itertools.combinations(range(n), r):
                if require_favorites and favorite_indices:
                    if not favorite_indices.issubset(combo):
                        continue
                total = sum(
                    records_list[i]["Cost_per_person_AUD"] * servings for i in combo
                )
                if min_total <= total <= budget and total > best_value:
                    best_value = total
                    best_combo = combo
        if best_combo is None:
            for r in range(1, n + 1):
                for combo in itertools.combinations(range(n), r):
                    if require_favorites and favorite_indices:
                        if not favorite_indices.issubset(combo):
                            continue
                    total = sum(
                        records_list[i]["Cost_per_person_AUD"] * servings for i in combo
                    )
                    if total <= budget and total > best_value:
                        best_value = total
                        best_combo = combo
        return best_combo, best_value

    best_subset, best_total = search(require_favorites=True)
    if best_subset is None:
        best_subset, best_total = search(require_favorites=False)
    if best_subset is None:
        best_subset = (0,)
    result = candidate_df.loc[list(best_subset)].reset_index(drop=True)
    result["Plan_Cost_AUD"] = result["Cost_per_person_AUD"] * servings
    result["Servings"] = servings
    result["Favorite"] = result["Recipe_Name"].isin(favorites)
    return result


def metric_figure(total_cost: float, budget: float, recipe_count: int, avg_score: float) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "indicator"} for _ in range(2)]],
        horizontal_spacing=0.15,
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=total_cost,
            number={"prefix": "$", "valueformat": ".2f"},
            title={"text": f"Total cost\n({recipe_count} recipes)"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=avg_score if not math.isnan(avg_score) else 0,
            title={"text": "Avg nutrition score"},
            number={"valueformat": ".0f", "suffix": " pts"},
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=15, r=15, t=30, b=15),
        height=160,
        font=dict(size=10),
    )
    return fig


def price_trend(selected: List[str]) -> go.Figure:
    if not selected:
        selected = []
    filtered = PRICE_DATA[PRICE_DATA["Ingredient_Name"].isin(selected)]
    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            annotations=[
                dict(
                    text="Select ingredients to view price trends.",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                )
            ],
        )
        return fig
    fig = px.line(
        filtered,
        x="Date",
        y="Price_AUD_per_unit",
        color="Ingredient_Name",
        markers=True,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Price (AUD per unit)",
        legend_title="Ingredient",
    )
    return fig





def substitution_suggestions(ingredient: str) -> List[dict]:
    base_price = INGREDIENT_AVG_PRICE.get(ingredient)

   
        
    suggestions = []
    for name, price in INGREDIENT_AVG_PRICE.items():
        if name == ingredient:
            continue
        
        if price < base_price:
            suggestions.append(
                {
                    "Ingredient": name.replace("_", " "),
                    "AveragePrice": round(price, 2),
                    "Saving": round(base_price - price, 2),
                }
            )
    return suggestions[:3]


app = Dash(__name__,
          external_stylesheets=[
              'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'
          ],
          external_scripts=[
              'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js'
          ])
server = app.server

app.layout = dcc.Tabs(
    id="main-tabs",
    value="tab-dashboard",
    className="dashboard-tabs",
    colors={"border": "#2563eb", "primary": "#1e293b", "background": "#e2e8f0"},
    children=[
        dcc.Tab(
            label=" Smart Cooking Dashboard",
            value="tab-dashboard",
            className="dashboard-tab",
            selected_className="dashboard-tab--selected",
            children=[
                dcc.Store(id="plan-store"),
                dcc.Loading(
                    id="dashboard-container",
                    className="dashboard-container",
                    color="#2563eb",
                    type="default",
                    children=[
                        html.Div(
                            className="container-fluid",
                            children=[
                                html.Div(
                                    className="row mb-3",
                                    children=[
                                        html.Div(
                                            className="col-12",
                                            children=[
                                                dcc.Markdown(
                                                    "Adjust the budget and dietary preferences to explore affordable meal plans.",
                                                    className="text-muted",
                                                    style={"fontSize": "14px"}
                                                )
    ]
)
                                    ]
                                ),
                                html.Div(
                                    className="row mb-3",
                                    children=[
                                        html.Div(
                                            className="col-md-8",
                                            children=[
                                                dcc.Markdown("**Budget (AUD)**", className="fw-bold"),
                                                dcc.RangeSlider(
                                                    id="budget-slider",
                                                    min=5,
                                                    max=100,
                                                    step=5,
                                                    value=[20, 60],
                                                    allowCross=False,
                                                    marks={i: f"${i}" for i in range(10, 101, 10)},
                                                    tooltip={"placement": "bottom", "always_visible": True},
                                                    className="mb-3",
                                                    updatemode="mouseup",
                                                    pushable=1,
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            className="col-md-4",
                                            children=[
                                                dcc.Markdown("**Servings**", className="fw-bold"),
                                                dcc.Slider(
                                                    id="serve-slider",
                                                    min=1,
                                                    max=6,
                                                    step=1,
                                                    value=DEFAULT_SERVINGS,
                                                    marks={i: str(i) for i in range(1, 7)},
                                                    tooltip={"placement": "bottom"},
                                                    className="mb-3",
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                html.Div(
                                    className="row mb-3",
                                    children=[
                                        html.Div(
                                            className="col-md-6",
                                            children=[
                                                dcc.Markdown("**Dietary preferences**", className="fw-bold"),
                                                dcc.Checklist(
                                                    id="dietary-checklist",
                                                    options=[{"label": diet, "value": diet} for diet in DIET_OPTIONS],
                                                    value=[],
                                                    inputStyle={"marginRight": "6px"},
                                                    className="mb-3",
                                                    style={"display": "flex", "flexWrap": "wrap", "gap": "10px"}
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            className="col-md-6",
                                            children=[
                                                dcc.Markdown("**Taste selection**", className="fw-bold"),
                                                dcc.Checklist(
                                                    id="taste-radio",
                                                    options=[{"label": label, "value": label} for label in TASTE_OPTIONS],
                                                    value=[],
                                                    inputStyle={"marginRight": "6px"},
                                                    className="mb-3",
                                                    style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                html.Div(
                                    className="row mb-3",
                                    children=[
                                        html.Div(
                                            className="col-12",
                                            children=[
                                                dcc.Markdown("**Favourite recipes**", className="fw-bold"),
                                                dcc.Dropdown(
                                                    id="favorite-dropdown",
                                                    options=[{"label": name, "value": name} for name in FAVORITE_OPTIONS],
                                                    value=[],
                                                    multi=True,
                                                    placeholder="Highlight preferred dishes",
                                                    className="mb-3",
                                                    style={"width": "100%"}
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                html.Div(
                                    className="row mb-3",
                                    children=[
                                        html.Div(
                                            className="col-12",
                                            children=[
                                                html.Div(
                                                    id="budget-caption",
                                                    className="alert alert-info py-2",
                                                    style={"fontSize": "14px"}
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                html.Div(
                                            className="col-6 col-md-8 col-lg-9",
                                            children=[
                                                dcc.Graph(
                                                    id="metric-graph",
                                                    config={"displayModeBar": False, "responsive": True},
                                                    className="card p-3",
                                                    style={"height": "180px", "marginBottom": "0", "width": "100%"}
                                                )
                                            ]
                                        ),
                                html.Div(
                                    className="row mb-4 d-flex align-items-stretch",
                                    style={"flexWrap": "nowrap"},
                                    children=[
                                        html.Div(
                                            className="col-6 col-md-4 col-lg-3",
                                            style={"maxWidth": "360px", "minWidth": "260px"},
                                            children=[
                                                html.Div(
                                                    className="card h-100",
                                                    children=[
                                                html.H5("Select Ingredients", className="mb-3"),
                                                html.Button(
                                                    "Clear Selection",
                                                    id="clear-ingredients",
                                                    n_clicks=0,
                                                    className="btn btn-outline-secondary btn-sm mb-3",
                                                ),
                                                dash_table.DataTable(
                                                    id="ingredient-table",
                                                    
                                                    columns=[
                                                        {"name": "", "id": "Image", "presentation": "markdown"},
                                                    ],
                                                    data=INGREDIENT_TABLE_DATA,
                                                    row_selectable="multi",
                                                    selected_rows=[],
                                                    style_table={
                                                        "height": "700px",
                                                        "overflowY": "auto",
                                                        "width": "150px",
                                                    },
                                                    style_header={
                                                        "backgroundColor": "#f8fafc",
                                                        "fontSize": "12px",
                                                        "fontWeight": "600",
                                                        "color": "#64748b",
                                                        "border": "0",
                                                        "height": "30px",
                                                        "padding": "4px 8px",
                                                    },
                                                    style_cell={
                                                        "padding": "2px",
                                                        "textAlign": "center",
                                                        "width": "10px",
                                                    },
                                                    style_cell_conditional=[
                                                        {"if": {"column_id": "Image"}, "width": "120px"},
                                                    ],
                                                    style_data_conditional=[
                                                        {
                                                            "if": {"state": "selected"},
                                                            "backgroundColor": "#dbeafe",
                                                            "border": "2px solid #2563eb",
                                                        }
                                                    ],
                                                ),
                                            ]
                                        ),
                                        
                                    ]
                                ),
                                html.Div(
                                    className="row mb-4",
                                    children=[
                                        html.Div(
                                            className="col-12",
                                            children=[
                                                html.Div(
                                                    className="card h-100",
                                                    children=[
                                                        dash_table.DataTable(
                                                            id="recipe-table",
                                                            columns=[
                                                                {"name": "Image", "id": "Image", "presentation": "markdown"},
                                                                {"name": "Recipe", "id": "Recipe"},
                                                                {"name": "Favorite", "id": "Favorite"},
                                                                {"name": "Cuisine", "id": "Cuisine"},
                                                                {"name": "Taste", "id": "Taste"},
                                                                {"name": "Cost per serve (AUD)", "id": "Cost"},
                                                                {"name": "Plan cost (AUD)", "id": "PlanCost"},
                                                                {"name": "Nutrition score", "id": "Nutrition"},
                                                                {"name": "Traffic light", "id": "Traffic"},
                                                                
                                                            ],
                                                            data=[],
                                                            style_table={
                                                                "height": "900px",
                                                                "overflowY": "auto",
                                                                'width': '1400px'
                                                            },
                                                            style_data_conditional=[
                                                                {
                                                                    "if": {"filter_query": "{Traffic} = 'Green'", "column_id": "Traffic"},
                                                                    "backgroundColor": "#22c55e",
                                                                    "color": "white",
                                                                },
                                                                {
                                                                    "if": {"filter_query": "{Traffic} = 'Amber'", "column_id": "Traffic"},
                                                                    "backgroundColor": "#fbbf24",
                                                                    "color": "black",
                                                                },
                                                                {
                                                                    "if": {"filter_query": "{Traffic} = 'Red'", "column_id": "Traffic"},
                                                                    "backgroundColor": "#f87171",
                                                                    "color": "black",
                                                                },
                                                                {
                                                                    "if": {"filter_query": "{Favorite} = '\u2605'"},
                                                                    "backgroundColor": "#fef3c7",
                                                                    "color": "#78350f",
                                                                },
                                                            ],
                                                            style_cell={
                                                                "padding": "8px",
                                                                "fontSize": "13px",
                                                                'width': '1400px'
                                                            },
                                                            style_header={
                                                                "backgroundColor": "#0f172a",
                                                                "color": "#f8fafc",
                                                                "fontWeight": "700",
                                                                "fontSize": "12px",
                                                            },
                                                        ),
                                                    ]
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                    ]
                                ),
                                html.Div(
                                    className="row mb-4",
                                    children=[
                                        html.Div(
                                            className="col-12",
                                            children=[
                                                html.H6("Ingredient Price Trend", className="mb-2"),
                                                dcc.Dropdown(
                                                    id="ingredient-multi",
                                                    options=INGREDIENT_OPTIONS,
                                                    multi=True,
                                                    value=[option["value"] for option in INGREDIENT_OPTIONS[:3]],
                                                    className="mb-3",
                                                    style={"width": "100%"}
                                                ),
                                                dcc.Graph(
                                                    id="price-trend",
                                                    className="card p-2",
                                                    style={"height": "200px"}
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                html.Div(
                                    className="row mb-4",
                                    children=[
                                        html.Div(
                                            className="col-md-6",
                                            children=[
                                                html.H6("Substitution", className="mb-2"),
                                                dcc.Dropdown(
                                                    id="substitution-dropdown",
                                                    options=INGREDIENT_OPTIONS,
                                                    value="Beef_mince",
                                                    className="mb-3",
                                                    style={"width": "100%"}
                                                ),
                                                dash_table.DataTable(
                                                    id="substitution-table",
                                                    columns=[
                                                        {"name": "Ingredient", "id": "Ingredient"},
                                                        {"name": "Average price (AUD)", "id": "AveragePrice"},
                                                        {"name": "Saving vs selected (AUD)", "id": "Saving"},
                                                    ],
                                                    data=[],
                                                    style_table={
                                                        "height": "150px",
                                                        "overflowY": "auto",
                                                    },
                                                    style_cell={
                                                        "padding": "6px",
                                                        "fontSize": "12px",
                                                    },
                                                    style_header={
                                                        "backgroundColor": "#0f172a",
                                                        "color": "#f8fafc",
                                                        "fontWeight": "600",
                                                        "fontSize": "11px",
                                                    },
                                                )
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        ],
                    ),
                ],
                ),
            ]
        )


@app.callback(
    Output("budget-caption", "children"),
    Output("metric-graph", "figure"),
    Output("recipe-table", "data"),
    Output("recipe-table", "tooltip_data"),
    Output("plan-store", "data"),
    Input("budget-slider", "value"),
    Input("serve-slider", "value"),
    Input("dietary-checklist", "value"),
    Input("taste-radio", "value"),
    Input("favorite-dropdown", "value"),
    Input("ingredient-table", "selected_rows"),
)
def update_plan(budget_range, servings, preferences, taste_choice, favorites, selected_ingredients):
    budget_range = budget_range or [20, 60]
    min_budget, max_budget = sorted(budget_range)
    servings = servings or DEFAULT_SERVINGS
    preferences = preferences or []
    taste_choice = taste_choice or "All tastes"
    favorites = favorites or []
    selected_ingredients = selected_ingredients or []

    # Convert selected row indices to ingredient names
    selected_ingredient_names = []
    if selected_ingredients:
        for idx in selected_ingredients:
            if idx < len(INGREDIENT_TABLE_DATA):              # Convert display name back to original ingredient name (replace spaces with underscores)
                display_name = INGREDIENT_TABLE_DATA[idx]["Ingredient"]
                original_name = display_name.replace(" ", "_")
                selected_ingredient_names.append(original_name)

    plan_df = plan_recipes(
        max_budget,
        preferences,
        min_budget,
        taste_choice,
        servings,
        favorites,
        selected_ingredient_names,
    )
    if plan_df.empty:
        ingredient_text = ""
        if selected_ingredient_names:
            ingredient_text = f" and selected ingredients ({', '.join(selected_ingredient_names)})"

        caption = (
            f"No recipes match the ${min_budget:.0f}-${max_budget:.0f} budget window "
            f"for {servings} serving(s) with the current filters{ingredient_text}."
        )
        empty_metrics = metric_figure(0, max_budget, 0, math.nan)
        return (
            caption,
            empty_metrics,
            [],
            [],
            {
                "plan": [],
                "budget_min": min_budget,
                "budget_max": max_budget,
                "total_cost": 0,
                "average_score": 0,
                "servings": servings,
                "favorites": favorites,
            },
        )
    total_cost = plan_df["Plan_Cost_AUD"].sum()
    avg_score = plan_df["Nutrition_Score"].mean()
    metric_fig = metric_figure(total_cost, max_budget, len(plan_df), avg_score)
    ingredient_text = ""
    if selected_ingredient_names:
        ingredient_text = f" using selected ingredients ({', '.join(selected_ingredient_names)})"

    within_range = min_budget <= total_cost <= max_budget
    if within_range:
        caption = (
            f"Planning {servings} serving(s) within a ${min_budget:.0f}-${max_budget:.0f} total budget "
            f"(current plan totals ${total_cost:.2f}){ingredient_text}."
        )
    else:
        relation = "below" if total_cost < min_budget else "above"
        caption = (
            f"Best available plan totals ${total_cost:.2f} for {servings} serving(s), which is {relation} the "
            f"${min_budget:.0f}-${max_budget:.0f} target range{ingredient_text}."
        )

    table_rows = []
    tooltips = []
    for _, row in plan_df.iterrows():
        price_name = row["Price_Name"]
        image_path = RECIPE_IMAGE_MAP.get(price_name)
        if image_path:
            src = dash.get_asset_url(image_path)
            image_markdown = f"![{row['Recipe_Name']}]({src})"
        else:
            image_markdown = "No image"
        score = row.get("Nutrition_Score", math.nan)
        traffic = traffic_light(score)
        table_rows.append(
            {
                "Image": image_markdown,
                "Recipe": row["Recipe_Name"],
                "Favorite": "★" if row.get("Favorite") else "",
                "Cuisine": row.get("Cuisine", ""),
                "Taste": row.get("Taste", ""),
                "Cost": f"${row['Cost_per_person_AUD']:.2f}",
                "PlanCost": f"${row['Plan_Cost_AUD']:.2f}",
                "Nutrition": f"{score:.0f}" if not math.isnan(score) else "N/A",
                "Traffic": traffic,
                "Ingredients": ", ".join(row["Ingredients"]),
            }
        )
        tooltips.append(
            {
                "Recipe": {"value": row.get("Description", "") or "No description", "type": "markdown"},
                "Ingredients": {
                    "value": "\n".join(f"- {item.replace('_', ' ')}" for item in row["Ingredients"]),
                    "type": "markdown",
                },
            }
        )

    plan_state = {
        "plan": plan_df.to_dict("records"),
        "budget_min": min_budget,
        "budget_max": max_budget,
        "total_cost": total_cost,
        "average_score": avg_score,
        "servings": servings,
        "favorites": favorites,
        "required_ingredients": selected_ingredient_names,
    }
    return caption, metric_fig, table_rows, tooltips, plan_state


@app.callback(Output("price-trend", "figure"), Input("ingredient-multi", "value"))
def update_trend(selected):
    return price_trend(selected or [])


@app.callback(
    Output("ingredient-table", "selected_rows"),
    Input("clear-ingredients", "n_clicks"),
    prevent_initial_call=True,
)
def clear_ingredient_selection(n_clicks):
    return []





@app.callback(Output("substitution-table", "data"), Input("substitution-dropdown", "value"))
def update_substitution(selected):
    if not selected:
        return []
    return substitution_suggestions(selected)


if __name__ == "__main__":
    app.run(debug=True)
