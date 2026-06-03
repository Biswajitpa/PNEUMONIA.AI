function initializeDiagnosticCharts(confidence, diagnosis) {
    const isPneumonia = diagnosis === "PNEUMONIA";
    const remainder = 100 - parseFloat(confidence);

    // 📊 1. PIE CHART GENERATION PIPELINE
    const pieCtx = document.getElementById('pieChart').getContext('2d');
    new Chart(pieCtx, {
        type: 'pie',
        data: {
            labels: ['Certainty Vector', 'Residual Delta'],
            datasets: [{
                data: [parseFloat(confidence), remainder],
                backgroundColor: isPneumonia ? ['#f43f5e', '#334155'] : ['#10b981', '#334155'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } }
        }
    });

    // 📈 2. BAR CHART BENCHMARK MAPPING PIPELINE
    const barCtx = document.getElementById('barChart').getContext('2d');
    new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: ['Train Mean', 'Active Check', 'Test Floor'],
            datasets: [{
                label: 'Accuracy Benchmark (%)',
                data: [84.64, parseFloat(confidence), 86.22],
                backgroundColor: ['#475569', '#38bdf8', '#64748b'],
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { 
                y: { display: false }, 
                x: { grid: { display: false } } 
            }
        }
    });
}