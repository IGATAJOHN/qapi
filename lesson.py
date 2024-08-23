import tkinter as tk  # Importing tkinter module for GUI

# Function to convert Celsius to Fahrenheit
def celsius_to_fahrenheit():
    try:
        # Get the Celsius value entered by the user
        celsius = float(entry_celsius.get())
        # Convert Celsius to Fahrenheit
        fahrenheit = (celsius * 9/5) + 32
        # Display the result in the Fahrenheit entry field
        entry_fahrenheit.delete(0, tk.END)
        entry_fahrenheit.insert(0, str(fahrenheit))
    except ValueError:
        # If the user input is not valid, show an error message
        entry_fahrenheit.delete(0, tk.END)
        entry_fahrenheit.insert(0, "Invalid Input")

# Function to convert Fahrenheit to Celsius
def fahrenheit_to_celsius():
    try:
        # Get the Fahrenheit value entered by the user
        fahrenheit = float(entry_fahrenheit.get())
        # Convert Fahrenheit to Celsius
        celsius = (fahrenheit - 32) * 5/9
        # Display the result in the Celsius entry field
        entry_celsius.delete(0, tk.END)
        entry_celsius.insert(0, str(celsius))
    except ValueError:
        # If the user input is not valid, show an error message
        entry_celsius.delete(0, tk.END)
        entry_celsius.insert(0, "Invalid Input")

# Create the main window
root = tk.Tk()
root.title("Temperature Converter")  # Set the window title

# Label for Celsius entry
label_celsius = tk.Label(root, text="Celsius")
label_celsius.grid(row=0, column=0)  # Position the label in the grid

# Entry field for Celsius input
entry_celsius = tk.Entry(root)
entry_celsius.grid(row=0, column=1)  # Position the entry field in the grid

# Label for Fahrenheit entry
label_fahrenheit = tk.Label(root, text="Fahrenheit")
label_fahrenheit.grid(row=1, column=0)  # Position the label in the grid

# Entry field for Fahrenheit input
entry_fahrenheit = tk.Entry(root)
entry_fahrenheit.grid(row=1, column=1)  # Position the entry field in the grid

# Button to convert Celsius to Fahrenheit
button_c_to_f = tk.Button(root, text="Celsius to Fahrenheit", command=celsius_to_fahrenheit)
button_c_to_f.grid(row=2, column=0, columnspan=2)  # Position the button in the grid

# Button to convert Fahrenheit to Celsius
button_f_to_c = tk.Button(root, text="Fahrenheit to Celsius", command=fahrenheit_to_celsius)
button_f_to_c.grid(row=3, column=0, columnspan=2)  # Position the button in the grid

# Run the Tkinter event loop (this keeps the window open)
root.mainloop()
