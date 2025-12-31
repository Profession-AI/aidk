def click_mouse(mouse_x: float, mouse_y: float, click_type: str = "right") -> None:
    """
    Use the computer to click the mouse at the given coordinates.

    Args:
        mouse_x (float): The x coordinate of the mouse normalized to the screen size (0.0 to 1.0)
        mouse_y (float): The y coordinate of the mouse normalized to the screen size (0.0 to 1.0)
        click_type (str): The type of click to perform. Can be "left" or "right", default is "right"
    """
    try:
        import pyautogui
    except ImportError:
        raise ImportError("pyautogui is not installed. Please install it with 'pip install pyautogui'")

    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    print(f"Screen dimensions: {screen_width}x{screen_height}")
    # Denormalize coordinates from 0.0-1.0 range to actual pixel coordinates
    actual_x = int(mouse_x * screen_width)
    actual_y = int(mouse_y * screen_height)
    
    # Ensure coordinates are within screen bounds
    actual_x = max(0, min(actual_x, screen_width - 1))
    actual_y = max(0, min(actual_y, screen_height - 1))

    if click_type == "left":
        pyautogui.click(actual_x, actual_y)
    elif click_type == "right":
        pyautogui.rightClick(actual_x, actual_y)
    else:
        raise ValueError(f"Invalid click type: {click_type}. Must be 'left' or 'right'")


def click_mouse_and_get_screenshot(mouse_x: float, mouse_y: float) -> dict:
    """
    Use the computer to click the mouse at the given coordinates and return the screenshot of the computer.

    Args:
        mouse_x (float): The x coordinate of the mouse normalized to the screen size (0.0 to 1.0)
        mouse_y (float): The y coordinate of the mouse normalized to the screen size (0.0 to 1.0)

    Returns:
        screenshot (PngImageFile): The screenshot of the computer
    """

    click_mouse(mouse_x, mouse_y)
    return {"image": _get_screenshot()}


def get_initial_screenshot() -> dict:
    """
    Get a screenshot of the computer in the initial state.

    Returns:
        screenshot (PngImageFile): The screenshot of the computer in the initial state
    """
    return {"image": _get_screenshot()}

def _get_screenshot():
    """
    Get a screenshot of the computer in the initial state.

    Returns:
        screenshot (PngImageFile): The screenshot of the computer in the initial state
    """
    try:
        import pyautogui
    except ImportError:
        raise ImportError("pyautogui is not installed. Please install it with 'pip install pyautogui'")

    return pyautogui.screenshot()   