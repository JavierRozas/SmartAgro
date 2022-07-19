d3.csv('https://raw.githubusercontent.com/JavierRozas/Prueba_Tesis/main/prueba.csv')
  .then(makeChart);



function makeChart(players) {
  var playerLabels = players.map(function(d) {return d.Temperatura_});
  var weeksData = players.map(function(d) {return d.Temperatura_});

  var chart = new Chart('chart', {
    type: 'line',
    data: {
      labels: [5.0, 6,0, 4.0],
      datasets: [
        {
          data: weeksData
        }
      ],
      backgroundColor: [
      'rgb(255, 99, 132)',
      'rgb(54, 162, 235)',
      'rgb(255, 205, 86)'
      ]
    }
  });
}