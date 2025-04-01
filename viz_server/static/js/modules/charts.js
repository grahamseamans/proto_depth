/**
 * Chart.js integration for visualization weight charts
 */

/**
 * Create weight charts for slot prototype weights
 */
function createWeightCharts(batchData) {
    if (!batchData || !batchData.metadata || !batchData.metadata.prototype_weights) {
        console.warn('No weight data available for charts');
        elements.weightChart.classList.add('hidden');
        return;
    }

    const weights = batchData.metadata.prototype_weights;
    if (!weights || weights.length === 0) {
        elements.weightChart.classList.add('hidden');
        return;
    }

    // Clear existing weights chart
    if (weightsChart) {
        weightsChart.destroy();
    }

    // Create datasets for the chart
    const numPrototypes = weights[0].length;
    const datasets = [];

    for (let slotIdx = 0; slotIdx < weights.length; slotIdx++) {
        const slotWeights = weights[slotIdx];
        const color = colors[slotIdx % colors.length];

        datasets.push({
            label: `Slot ${slotIdx + 1}`,
            data: slotWeights,
            backgroundColor: `#${color.toString(16).padStart(6, '0')}`,
            borderColor: 'rgba(0, 0, 0, 0.2)',
            borderWidth: 1,
        });
    }

    // Create labels for prototypes
    const labels = Array.from({ length: numPrototypes }, (_, i) => `Proto ${i + 1}`);

    // Initialize Chart.js
    const ctx = document.getElementById('weights-chart').getContext('2d');
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Prototype Weights per Slot'
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const value = context.raw;
                            return `${context.dataset.label}: ${value.toFixed(3)}`;
                        }
                    }
                }
            }
        }
    });

    // Store chart reference
    setWeightsChart(chart);
}
