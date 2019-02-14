function openPage(pageName,elmnt,color) {
	var i, tabcontent, tablinks;
	tabcontent = document.getElementsByClassName("tabcontent");
	for (i = 0; i < tabcontent.length; i++) {
	tabcontent[i].style.display = "none";
	}
	tablinks = document.getElementsByClassName("tablink");
	for (i = 0; i < tablinks.length; i++) {
	tablinks[i].style.backgroundColor = "";
	}
	document.getElementById(pageName).style.display = "block";
	elmnt.style.backgroundColor = color;
}

//parameters
function chooseDemand(demandType){
	demand=demandType;
};
function chooseAgent(agentType){
	agent=agentType;
};

//charts
function agent_chart(chartName){
	var newChart = new Chart(document.getElementById(chartName), 
	{
	type: "bar",
	data: {
		labels: [],
		datasets:[{ 
			data: [0], 
			backgroundColor: "#3e95cd"}]},
	options: {
		legend: {display: false},
		tooltips: {enabled: false},
		scales: {
			yAxes: [{
				ticks: {min: -10, max: 100}
			}]}}
	})
	return newChart;
}
function cost_chart(chartName,legAgent, legAI){
	var newChart = new Chart(document.getElementById(chartName), 
	{
	type: 'line',
	data: 
	{
		labels: [],
		datasets:
		[
		{ 	data: [],
			label: legAgent,
			borderColor: "#3e95cd",
			fill: false },
		{ 	data: [],
			label: legAI,
			borderColor: "#FF0000",
			fill: false }
		]
	  }
	})
	return newChart;
}

//simulation
async function runSimulation(results,results2){
	//results = {retailerStock, wholesalerStock, distributorStock, factoryStock}
	removeAllData(costChart);
	for (i = 0; i < results[0].length ; i++){
		var time = document.getElementById("myRange").value;
		addData(costChart, i, results[0][i] + results[1][i] + results[2][i] + results[3][i], (results2[0][i] + results2[1][i] + results2[2][i] + results2[3][i]));

		updateData(retailerChart, results[0][i]);
		verifyStock(results[0][i], "retailerArrow");
		updateData(wholesalerChart, results[1][i])
		verifyStock(results[1][i], "wholesalerArrow");
		updateData(distributorChart, results[2][i]);
		verifyStock(results[2][i], "distributorArrow");
		updateData(factoryChart, results[3][i]);
		verifyStock(results[3][i], "factoryArrow");
		await sleep(time);
	};
	isRunning = false;
};
function getData(button_selection){
    if (isRunning == false){   
	    isRunning = true;
	    if (button_selection==4){
	    	var taux = document.getElementById("TS");
			var strTaux = taux.options[taux.selectedIndex].value;

			var leadtime = document.getElementById("lt");
			var strleadtime = leadtime.options[leadtime.selectedIndex].value;

			var list_vars = [demand, strTaux, strleadtime, agent];
	    }
	    else if (button_selection==1){
	    	var list_vars = ["Gaussian", "99", '1', "3"];
	    }
	    else if (button_selection==2){
	    	var list_vars = ["Gaussian", '95', '1', "4"];
	    }
	    else if (button_selection==3){
	    	var list_vars = ["Seasonal", '95', '1', "4"];
	    }
	    else{
	    	var list_vars = ["Gaussian", '95', '1', "1"];
	    }

		var parameters_list = JSON.stringify(list_vars);

	    $.post("/get_data", parameters_list, function(data){
			var usabledata_all = JSON.parse(data);
			var usabledata = usabledata_all[0];
			usabledata2 = usabledata_all[1];
			runSimulation(usabledata,usabledata2);
		});
	};
	//alternative method to send data to the server
	/*$.ajax({
		url: "/test_json",
		type: "POST",
		contentType:"applicaton/json",
		data: Json_list,
		success: function(serverData){
			alert(serverData);
		}
		});*/
};
function addData(chart, label, firstData, secondData) {
	chart.data.labels.push(label);
	chart.data.datasets[0].data.push(firstData);
	chart.data.datasets[1].data.push(secondData);
	chart.update();
}
function removeAllData(chart) {
    while (chart.data.labels[0] != null){
	    chart.data.labels.pop();
	    chart.data.datasets.forEach((dataset) => {
	        dataset.data.pop();
	    });}
	chart.update();
}
function updateData(chart, data) {
	chart.data.datasets[0].data[0] = data;
	chart.update();
}
function verifyStock(stock, imageId) {
	if (stock <= 0){	document.getElementById(imageId).src = "/static/img/RedArrow.png";}
	else { document.getElementById(imageId).src = "/static/img/GreenArrow.png";}
}
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


