# Why Streamlit?

## Why we chose Streamlit
Streamlit is a strong choice for this project because it is fast to build, easy to use, and well suited for prototyping machine learning GUIs.
It lets the team focus on the handwritten digit recognition workflow instead of frontend details like HTML/CSS or JavaScript frameworks.

Key reasons:
- Minimal setup and rapid development.
- Built-in support for image display, sliders, buttons, and layout.
- Easy integration with Python preprocessing and model code.
- Good for non-technical users and quick demos.

## Comparison with other GUI options

### Streamlit
- Best for Python-first prototypes.
- Simple syntax and fast iteration.
- Good default UI for data apps.
- No need to write separate frontend/backend code.

### Flask / Django
- More flexible for custom web applications.
- Requires HTML templates, CSS, and possibly JavaScript.
- Better for production-grade websites, but slower to prototype.
- Not ideal for a quick ML demo in a course project.

### Gradio
- Also easy for machine learning demos and model interaction.
- Excellent for sharing standalone model demos.
- Less flexible for custom UI layout than Streamlit.
- Streamlit was chosen because it provides a stronger layout system and sidebar controls.

### Dash / Plotly Dash
- Powerful for interactive dashboards and data visualization.
- More setup and boilerplate than Streamlit.
- Better when complex callbacks or interactive charts are required.
- Overkill for this simple handwritten digit recognition GUI.

### Tkinter / PyQt
- Desktop GUI libraries, not web-based.
- Requires more UI programming and is less shareable over the web.
- Not as convenient for a course project that should run in a browser.

## Conclusion
Streamlit is the best fit for this project because it enables a user-friendly browser interface with minimal development overhead, while still supporting the preprocessing controls and image visualization needed for the GUI.
