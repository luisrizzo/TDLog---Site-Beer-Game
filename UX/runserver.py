from flask import Markup
from flask import Flask
from flask import Markup
from flask import Flask
from flask import render_template
from flask import request, redirect
import time
import simulation as sim
import json

import pickle

app = Flask(__name__)

@app.route("/")
def load_main():
	return render_template('index.html')

@app.route("/simulation")
def simulation():
	return render_template('simulation.html')

# Function that send the data to the client side 
@app.route('/get_data', methods = ['POST'])
def get_data():

	parameters = request.get_json(force=True)
	demand_type = parameters[0]
	TS = int(parameters[1])/100
	lt = int(parameters[2])
	agentpos = int(parameters[3])-1
	print(agentpos)
	history_BS = sim.get_history(demand_type,TS,lt, agentpos, False)
	hIL =[]
	indexes = [1,2,3,4]
	for i in indexes :
		hIL.append(history_BS['IL'][:,i].tolist())
	#data_to_send = json.dumps(hIL)
	
	TS = max(TS,0.95)
	demand_type = "Seasonal"

	key_dict = ",".join(map(str,(lt,',',agentpos,',',demand_type,',',TS)))
	data_part_2 = open_database(key_dict)

	data_to_send = json.dumps((hIL,data_part_2))
	return data_to_send

@app.route('/signup', methods = ['POST'])
def signup():
	email = request.form['email']
	save_email(email)
	return redirect('/')

def open_database(key_dict):
	pickle_off = open("history.pickle","rb")
	history_AI = pickle.load(pickle_off)
	hIL_database =[]
	indexes = [1,2,3,4]
	for i in indexes :
		hIL_database.append(history_AI[key_dict]['IL'][:,i].tolist())
	return hIL_database

def save_email(email_adress):
	pickle_on_mail = open("mail.pickle","wb")
	email_addresses = pickle.load(pickle_off)
	email_addresses.append(email_adress)
	pickle.dump(email_addresses,pic)
	pickle_on_mail.close()

if __name__ == "__main__":
	app.run()	

'''
def contact():
	if request.method == 'POST':
		if request.form['submit_button'] == 1:
			pass # do something
		elif request.form['submit_button'] == 'Do Something Else':
			pass # do something else
		else:
			pass # unknown
	elif request.method == 'GET':
		return render_template('contact.html', form=form)

@app.route('/test_json', methods = ['POST'])
def test_json():
	clientData = request.get_json(force=True)
	print(clientData[0])
	data_to_send2 = get_data()
	return data_to_send2
'''
