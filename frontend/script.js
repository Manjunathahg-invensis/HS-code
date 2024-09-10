async function displayOutput() {
    const inputText = document.getElementById('inputText').value;

    // Show loading spinner
    document.getElementById('loading').style.display = 'block';

    // Send POST request to the FastAPI endpoint
    try {
        const response = await fetch('http://127.0.0.1:8000/query/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: inputText }),
        });

        if (response.ok) {
            const result = await response.json();

            // Display the result
            sessionStorage.setItem('result', JSON.stringify(result));
            showResult(result);
        } else {
            alert(`Error: ${response.statusText}`);
        }
    } catch (error) {
        console.error('Error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        // Hide the loading indicator
        document.getElementById('loading').style.display = 'none'; // Hide loading spinner
    }
}

function showResult(result) {
    const outputDiv = document.getElementById('output');

    // Create table structure if data exists
    if (result && result.length > 0) {
        let table = `<table>
                        <tr>
                            <th>HS Code</th>
                            <th>Score</th>
                            <th>Chapter Description</th>
                            <th>Description</th>
                        </tr>`;
        
        result.forEach(item => {
            table += `<tr>
                        <td>${item['HS Code']}</td>
                        <td>${item['Score']}%</td>
                        <td>${item['chap_desc']}</td>
                        <td>${item['desc']}</td>
                      </tr>`;
        });

        table += `</table>`;
        outputDiv.innerHTML = table;
    } else {
        outputDiv.innerHTML = `<p>No Data Found</p>`;
    }

    // Switch visibility to output section
    document.getElementById('inputSection').style.display = 'none';
    document.getElementById('outputSection').style.display = 'block';
}

function goBack() {
    location.reload(); // Refresh the page
}
