<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $fullname = $_POST['fullname'] ?? '';
    $position = $_POST['position'] ?? '';

    if (isset($_FILES['resume']) && $_FILES['resume']['error'] === UPLOAD_ERR_OK) {
        $fileTmpPath = $_FILES['resume']['tmp_name'];
        $fileName = $_FILES['resume']['name'];

        // �������������� ������ ��� REST API
        $postData = [
            'fullname' => $fullname,
            'position' => $position,
            'resume'   => new CURLFile($fileTmpPath, 'application/pdf', $fileName)
        ];

        $ch = curl_init('http://192.168.100.203/api/receive');
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $postData);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);

        if (curl_errno($ch)) {
            echo "������ ��� ����������: " . curl_error($ch);
        } else {
            if ($httpCode === 200) {
                echo "������ ������� ����������!";
            } else {
                echo "������ API: HTTP $httpCode <br> �����: $response";
            }
        }
        curl_close($ch);
    } else {
        echo "������ �������� �����.";
    }
} else {
    echo "�������� ����� �������.";
}
?>
