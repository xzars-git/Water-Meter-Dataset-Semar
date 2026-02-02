"""
Custom Gradio Theme for Water Meter AI Detection System
========================================================
A Clean Light Corporate theme with professional navy blue accents.

CRITICAL: LIGHT THEME ONLY
- Background: #F9F7F7 (Off-White) - ENFORCED
- Cards/Blocks: #FFFFFF (Pure White)
- Text: #112D4E (Dark Navy) - NOT white
- Accent: #3F72AF (Medium Blue)
"""

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


class SeaSideTheme(Base):
    """
    SeaSide Theme - Clean Light Corporate Design
    
    A professional light theme for the Water Meter AI Detection System.
    Enforces light background with Navy Blue brand accents.
    """
    
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.blue,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.slate,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_lg,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | tuple[fonts.Font | str, ...] = (
            fonts.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | tuple[fonts.Font | str, ...] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ),
    ):
        # Define Navy Blue as primary color
        primary_hue = gr.themes.Color(
            c50="#f0f4f8",
            c100="#d9e2ec",
            c200="#bcccdc",
            c300="#9fb3c8",
            c400="#829ab1",
            c500="#3F72AF",   # Medium Blue
            c600="#2e5a8f",
            c700="#1f4470",
            c800="#112D4E",   # Navy Blue (PRIMARY)
            c900="#0d2238",
            c950="#081624",
            name="primary",
        )
        
        # Secondary: Medium Blue
        secondary_hue = gr.themes.Color(
            c50="#f0f6fb",
            c100="#d9e8f5",
            c200="#b3d1eb",
            c300="#8dbae1",
            c400="#67a3d7",
            c500="#3F72AF",   # Medium Blue (SECONDARY)
            c600="#355f92",
            c700="#2b4c75",
            c800="#213958",
            c900="#17263b",
            c950="#0e1623",
            name="secondary",
        )
        
        # Neutral: Light grays for light theme
        neutral_hue = gr.themes.Color(
            c50="#F9F7F7",    # Off-White (BACKGROUND)
            c100="#f5f6f8",
            c200="#DBE2EF",   # Light Blue/Grey
            c300="#c8d1e0",
            c400="#b5c0d1",
            c500="#a2afc2",
            c600="#8f9eb3",
            c700="#7c8da4",
            c800="#112D4E",   # Dark Navy for text
            c900="#566b86",
            c950="#435a77",
            name="neutral",
        )
        
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        
        # CRITICAL: Force Light Theme (Disable Dark Mode behavior completely)
        super().set(
            # --- BASE LAYERS ---
            # We map _dark variables to the exact same values as light variables
            # This prevents Gradio from switching to black background in Dark Mode
            body_background_fill="#F9F7F7",
            body_background_fill_dark="#F9F7F7", 
            
            body_text_color="#112D4E",
            body_text_color_dark="#112D4E",
            
            body_text_color_subdued="#64748b",
            body_text_color_subdued_dark="#64748b",

            # --- BLOCKS / CARDS ---
            block_background_fill="#FFFFFF",
            block_background_fill_dark="#FFFFFF",
            
            block_border_color="#DBE2EF",
            block_border_color_dark="#DBE2EF",
            
            block_border_width="1px",
            
            block_label_background_fill="#FFFFFF",
            block_label_background_fill_dark="#FFFFFF",
            
            block_label_text_color="#112D4E",
            block_label_text_color_dark="#112D4E",
            
            block_padding="*spacing_lg",
            block_radius="*radius_lg",
            block_shadow="0 4px 6px rgba(17, 45, 78, 0.05)",
            
            block_title_text_color="#FFFFFF",
            block_title_text_color_dark="#FFFFFF",
            
            block_title_background_fill="#112D4E",
            block_title_background_fill_dark="#112D4E",

            # --- BUTTONS ---
            button_primary_background_fill="#112D4E",
            button_primary_background_fill_dark="#112D4E",
            
            button_primary_text_color="#FFFFFF",
            button_primary_text_color_dark="#FFFFFF",
            
            button_primary_border_color="#112D4E",
            button_primary_border_color_dark="#112D4E",

            button_secondary_background_fill="#3F72AF",
            button_secondary_background_fill_dark="#3F72AF",
            
            button_secondary_text_color="#FFFFFF",
            button_secondary_text_color_dark="#FFFFFF",

            # --- INPUTS ---
            input_background_fill="#FFFFFF",
            input_background_fill_dark="#FFFFFF",
            
            input_border_color="#DBE2EF",
            input_border_color_dark="#DBE2EF",
            
            input_placeholder_color="#94a3b8",
            input_placeholder_color_dark="#94a3b8",

            # --- PANELS ---
            panel_background_fill="#FFFFFF",
            panel_background_fill_dark="#FFFFFF",
            
            panel_border_color="#DBE2EF",
            panel_border_color_dark="#DBE2EF",

            # --- TABLES ---
            table_border_color="#DBE2EF",
            table_border_color_dark="#DBE2EF",
            
            table_even_background_fill="#F9F7F7",
            table_even_background_fill_dark="#F9F7F7",
            
            table_odd_background_fill="#FFFFFF",
            table_odd_background_fill_dark="#FFFFFF",
        )


# Convenience function to get the theme instance
def get_seaside_theme() -> SeaSideTheme:
    """
    Returns an instance of the SeaSide Light Corporate theme.
    
    Returns:
        SeaSideTheme: Configured light theme instance
    """
    return SeaSideTheme()


# Example usage for testing
if __name__ == "__main__":
    print("=" * 60)
    print("SeaSide Theme - Clean Light Corporate")
    print("=" * 60)
    print("✅ Light Background: #F9F7F7 (Off-White)")
    print("✅ White Cards: #FFFFFF (Pure White)")
    print("✅ Navy Accent: #112D4E (Dark Navy)")
    print("✅ Text Color: #112D4E (Dark on Light)")
    print("=" * 60)
    print("\nTheme created successfully!")
    print("Usage: theme = get_seaside_theme()")

