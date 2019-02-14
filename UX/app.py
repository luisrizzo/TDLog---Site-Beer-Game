from flask import Flask
from flask import Markup
from flask import Flask
from flask import render_template
from flask import request, redirect
import time
app = Flask(__name__)
values = [10,9,8,7,6,5,4,3,2,1,0]
i = 0
labels = ["Graphique","Constant"]

@app.route("/")
def chart():
	global labels
	global values
	global i
	send = (values[i],values[i+1])
	return render_template('chart.html', values=send, labels=labels)


@app.route('/signup', methods = ['POST'])
def signup():
	global i
	i+=1
	return redirect('/')
 
@app.route('/chart2')
def page2():
	global i
	i+=1
	return render_template('chart2.html')

# Function that send the data to the client side 
@app.route('/get_data35', methods = ['GET'])
def get_data():
	data_to_send = 'Hello World From the Server !!'
	return data_to_send

if __name__ == "__main__":
	app.run()