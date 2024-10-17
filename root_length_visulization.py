# import pandas as pd
# import matplotlib.pyplot as plt
# from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
# from PyQt6.QtCore import Qt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# class RootLengthVisualization(QWidget):
#     def __init__(self, csv_path):
#         super().__init__()
#         print(f"DEBUG: Initializing RootLengthVisualization with csv_path: {csv_path}")
#         self.setWindowTitle("Root Length Visualization")
#         # self.setGeometry(1500, 1500, 1000, 700)  # Increased window size
#         self.showFullScreen()
#         self.csv_path = csv_path
#         self.df = pd.read_csv(csv_path)
#         print(f"DEBUG: CSV file loaded. Shape: {self.df.shape}")

#         self.df["Date"] = pd.to_datetime(
#             self.df["Image"].str.extract(r"(\d{4}\.\d{2}\.\d{2})")[0]
#         )
#         self.df["Tube"] = self.df["Image"].str.extract(r"T(\d+)")[0].astype(int)
#         self.df["Position"] = self.df["Image"].str.extract(r"L(\d+)")[0].astype(int)
#         print("DEBUG: Date, Tube, and Position columns extracted")
#         print(f"DEBUG: Unique Tubes: {self.df['Tube'].unique()}")
#         print(f"DEBUG: Unique Positions: {self.df['Position'].unique()}")

#         self.init_ui()

#     def init_ui(self):
#         print("DEBUG: Initializing UI")
#         main_layout = QVBoxLayout()
#         self.setLayout(main_layout)

#         # Create a widget to hold the plot and center it
#         plot_widget = QWidget()
#         plot_layout = QVBoxLayout(plot_widget)
#         main_layout.addWidget(plot_widget, alignment=Qt.AlignmentFlag.AlignCenter)

#         self.fig, self.ax = plt.subplots(figsize=(10, 7))  # Increased figure size
#         self.canvas = FigureCanvas(self.fig)
#         plot_layout.addWidget(self.canvas)

#         # Info label with increased font size
#         self.info_label = QLabel()
#         self.info_label.setStyleSheet("font-size: 14px; margin: 10px;")
#         self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         main_layout.addWidget(self.info_label)

#         # Back button
#         button_layout = QHBoxLayout()
#         main_layout.addLayout(button_layout)

#         self.back_button = QPushButton("Back")
#         self.back_button.setStyleSheet("font-size: 14px; padding: 5px 10px;")
#         self.back_button.clicked.connect(self.go_back)
#         button_layout.addWidget(
#             self.back_button, alignment=Qt.AlignmentFlag.AlignCenter
#         )
#         self.back_button.hide()

#         self.current_view = "tubes"
#         self.current_tube = None
#         self.current_position = None
#         self.current_cid = None

#         self.show_tubes()

#     def show_tubes(self):
#         print("DEBUG: Showing tubes")
#         self.ax.clear()
#         tube_lengths = self.df.groupby("Tube")["Length (mm)"].sum()
#         print(f"DEBUG: Tube lengths: {tube_lengths}")
#         bars = tube_lengths.plot(kind="bar", ax=self.ax)
#         self.ax.set_title("Combined Root Length per Tube", fontsize=16)
#         self.ax.set_xlabel("Tube Number", fontsize=12)
#         self.ax.set_ylabel("Total Length (mm)", fontsize=12)
#         self.ax.tick_params(axis="both", which="major", labelsize=10)

#         # Add value labels on top of each bar
#         for bar in bars.patches:
#             self.ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 bar.get_height(),
#                 f"{bar.get_height():.2f}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=10,
#             )

#         plt.tight_layout()
#         self.canvas.draw()
#         self.info_label.setText("Click on a bar to see length per section")
#         self.back_button.hide()
#         self.disconnect_event()
#         self.current_cid = self.canvas.mpl_connect(
#             "button_press_event", self.on_tube_click
#         )
#         self.current_view = "tubes"

#     def show_sections(self, tube):
#         print(f"DEBUG: Showing sections for tube {tube}")
#         self.ax.clear()
#         section_lengths = (
#             self.df[self.df["Tube"] == tube].groupby("Position")["Length (mm)"].mean()
#         )
#         print(f"DEBUG: Section lengths for tube {tube}: {section_lengths}")
#         bars = section_lengths.plot(kind="bar", ax=self.ax)
#         self.ax.set_title(f"Root Length per Section in Tube {tube}", fontsize=16)
#         self.ax.set_xlabel("Position within Tube", fontsize=12)
#         self.ax.set_ylabel("Average Length (mm)", fontsize=12)
#         self.ax.tick_params(axis="both", which="major", labelsize=10)

#         # Add value labels on top of each bar
#         for bar in bars.patches:
#             self.ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 bar.get_height(),
#                 f"{bar.get_height():.2f}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=10,
#             )

#         plt.tight_layout()
#         self.canvas.draw()
#         self.info_label.setText("Click on a bar to see length changes over time")
#         self.back_button.show()
#         self.disconnect_event()
#         self.current_cid = self.canvas.mpl_connect(
#             "button_press_event", self.on_section_click
#         )
#         self.current_view = "sections"
#         self.current_tube = tube

#     def show_time_series(self, tube, position):
#         print(f"DEBUG: Showing time series for tube {tube}, position {position}")
#         self.ax.clear()
#         time_series = self.df[
#             (self.df["Tube"] == tube) & (self.df["Position"] == position)
#         ].set_index("Date")["Length (mm)"]
#         print(f"DEBUG: Time series data: {time_series}")
#         time_series.plot(ax=self.ax, marker="o")
#         self.ax.set_title(
#             f"Root Length Changes Over Time (Tube {tube}, Position {position})",
#             fontsize=16,
#         )
#         self.ax.set_xlabel("Date", fontsize=12)
#         self.ax.set_ylabel("Length (mm)", fontsize=12)
#         self.ax.tick_params(axis="both", which="major", labelsize=10)
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         self.canvas.draw()
#         self.info_label.setText("")
#         self.back_button.show()
#         self.disconnect_event()
#         self.current_view = "time_series"
#         self.current_tube = tube
#         self.current_position = position

#     def on_tube_click(self, event):
#         if event.inaxes == self.ax:
#             tube = int(round(event.xdata)) + 1
#             print(f"DEBUG: Tube {tube} clicked")
#             self.show_sections(tube)

#     def on_section_click(self, event):
#         if event.inaxes == self.ax:
#             position = int(round(event.xdata)) + 1
#             print(f"DEBUG: Section {position} clicked for tube {self.current_tube}")
#             self.show_time_series(self.current_tube, position)

#     def go_back(self):
#         if self.current_view == "sections":
#             self.show_tubes()
#         elif self.current_view == "time_series":
#             self.show_sections(self.current_tube)

#     def disconnect_event(self):
#         if self.current_cid is not None:
#             self.canvas.mpl_disconnect(self.current_cid)
#             self.current_cid = None


import sys
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State
import dash
import dash_bootstrap_components as dbc
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl


class RootLengthVisualization(QMainWindow):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        self.df["Date"] = pd.to_datetime(
            self.df["Image"].str.extract(r"(\d{4}\.\d{2}\.\d{2})")[0]
        )
        self.df["Tube"] = self.df["Image"].str.extract(r"T(\d+)")[0].astype(int)
        self.df["Position"] = self.df["Image"].str.extract(r"L(\d+)")[0].astype(int)

        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_dash_layout()
        self.setup_dash_callbacks()

        self.setup_pyqt_ui()

    def setup_dash_layout(self):
        self.app.layout = dbc.Container(
            [
                html.H1("Root Length Visualization", className="text-center my-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id="main-graph"),
                                html.Div(id="click-data", className="text-center my-2"),
                                dbc.Button(
                                    "Back", id="back-button", className="mt-2 d-none"
                                ),
                            ]
                        )
                    ]
                ),
            ],
            fluid=True,
        )

    def setup_dash_callbacks(self):
        @self.app.callback(
            [
                Output("main-graph", "figure"),
                Output("click-data", "children"),
                Output("back-button", "className"),
            ],
            [Input("main-graph", "clickData"), Input("back-button", "n_clicks")],
            [State("main-graph", "figure")],
        )
        def update_graph(clickData, n_clicks, current_figure):
            ctx = dash.callback_context
            if not ctx.triggered:
                return (
                    self.show_tubes(),
                    "Click on a bar to see length per section",
                    "mt-2 d-none",
                )

            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if button_id == "back-button":
                if current_figure["layout"]["title"]["text"].startswith(
                    "Root Length Changes Over Time"
                ):
                    tube = int(
                        current_figure["layout"]["title"]["text"]
                        .split("Tube ")[1]
                        .split(",")[0]
                    )
                    return (
                        self.show_sections(tube),
                        f"Showing sections for Tube {tube}",
                        "mt-2",
                    )
                else:
                    return (
                        self.show_tubes(),
                        "Click on a bar to see length per section",
                        "mt-2 d-none",
                    )

            if not clickData:
                return dash.no_update, dash.no_update, dash.no_update

            click_value = clickData["points"][0]["x"]
            if (
                current_figure["layout"]["title"]["text"]
                == "Combined Root Length per Tube"
            ):
                tube = int(click_value)
                return (
                    self.show_sections(tube),
                    f"Click on a bar to see length changes over time for Tube {tube}",
                    "mt-2",
                )
            elif current_figure["layout"]["title"]["text"].startswith(
                "Root Length per Section"
            ):
                tube = int(current_figure["layout"]["title"]["text"].split("Tube ")[1])
                position = int(click_value)
                return self.show_time_series(tube, position), "", "mt-2"

            return dash.no_update, dash.no_update, dash.no_update

    def setup_pyqt_ui(self):
        self.setWindowTitle("Root Length Visualization")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)

        # Start the Dash app in a separate thread
        from threading import Thread

        self.thread = Thread(target=self.run_dash_app)
        self.thread.daemon = True
        self.thread.start()

        # Load the Dash app in the QWebEngineView
        self.web_view.load(QUrl("http://localhost:8050"))

    def run_dash_app(self):
        self.app.run_server(debug=False, port=8050)

    def show_tubes(self):
        tube_lengths = self.df.groupby("Tube")["Length (mm)"].sum().reset_index()
        fig = go.Figure(
            data=[
                go.Bar(
                    x=tube_lengths["Tube"],
                    y=tube_lengths["Length (mm)"],
                    text=tube_lengths["Length (mm)"].round(2),
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Combined Root Length per Tube",
            xaxis_title="Tube Number",
            yaxis_title="Total Length (mm)",
            clickmode="event+select",
        )
        return fig

    def show_sections(self, tube):
        section_lengths = (
            self.df[self.df["Tube"] == tube]
            .groupby("Position")["Length (mm)"]
            .mean()
            .reset_index()
        )
        fig = go.Figure(
            data=[
                go.Bar(
                    x=section_lengths["Position"],
                    y=section_lengths["Length (mm)"],
                    text=section_lengths["Length (mm)"].round(2),
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title=f"Root Length per Section in Tube {tube}",
            xaxis_title="Position within Tube",
            yaxis_title="Average Length (mm)",
            clickmode="event+select",
        )
        return fig

    def show_time_series(self, tube, position):
        time_series = self.df[
            (self.df["Tube"] == tube) & (self.df["Position"] == position)
        ].sort_values("Date")
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=time_series["Date"],
                    y=time_series["Length (mm)"],
                    mode="lines+markers",
                )
            ]
        )
        fig.update_layout(
            title=f"Root Length Changes Over Time (Tube {tube}, Position {position})",
            xaxis_title="Date",
            yaxis_title="Length (mm)",
        )
        return fig
