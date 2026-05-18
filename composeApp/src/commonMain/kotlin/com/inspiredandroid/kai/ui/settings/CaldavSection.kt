package com.inspiredandroid.kai.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.inspiredandroid.kai.ui.KaiOutlinedTextField
import com.inspiredandroid.kai.ui.handCursor
import kai.composeapp.generated.resources.Res
import kai.composeapp.generated.resources.settings_caldav
import kai.composeapp.generated.resources.settings_caldav_description
import kai.composeapp.generated.resources.settings_caldav_password
import kai.composeapp.generated.resources.settings_caldav_save
import kai.composeapp.generated.resources.settings_caldav_test
import kai.composeapp.generated.resources.settings_caldav_test_success
import kai.composeapp.generated.resources.settings_caldav_testing
import kai.composeapp.generated.resources.settings_caldav_url
import kai.composeapp.generated.resources.settings_caldav_url_hint
import kai.composeapp.generated.resources.settings_caldav_username
import org.jetbrains.compose.resources.stringResource

@Composable
internal fun CaldavSection(
    savedUrl: String,
    savedUsername: String,
    savedPassword: String,
    testStatus: CaldavTestStatus,
    onSave: (url: String, username: String, password: String) -> Unit,
    onTest: (url: String, username: String, password: String) -> Unit,
) {
    var url by remember(savedUrl) { mutableStateOf(savedUrl) }
    var username by remember(savedUsername) { mutableStateOf(savedUsername) }
    var password by remember(savedPassword) { mutableStateOf(savedPassword) }

    Column(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = stringResource(Res.string.settings_caldav),
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(Modifier.height(4.dp))
        Text(
            text = stringResource(Res.string.settings_caldav_description),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(12.dp))

        KaiOutlinedTextField(
            value = url,
            onValueChange = { url = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text(stringResource(Res.string.settings_caldav_url)) },
            placeholder = { Text(stringResource(Res.string.settings_caldav_url_hint)) },
            singleLine = true,
        )
        Spacer(Modifier.height(8.dp))

        KaiOutlinedTextField(
            value = username,
            onValueChange = { username = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text(stringResource(Res.string.settings_caldav_username)) },
            singleLine = true,
        )
        Spacer(Modifier.height(8.dp))

        KaiOutlinedTextField(
            value = password,
            onValueChange = { password = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text(stringResource(Res.string.settings_caldav_password)) },
            visualTransformation = PasswordVisualTransformation(),
            singleLine = true,
        )
        Spacer(Modifier.height(12.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(
                onClick = { onSave(url, username, password) },
                modifier = Modifier.handCursor(),
            ) {
                Text(stringResource(Res.string.settings_caldav_save))
            }
            OutlinedButton(
                onClick = { onTest(url, username, password) },
                enabled = testStatus !is CaldavTestStatus.Testing,
                modifier = Modifier.handCursor(),
            ) {
                if (testStatus is CaldavTestStatus.Testing) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                } else {
                    Text(stringResource(Res.string.settings_caldav_test))
                }
            }
        }

        when (testStatus) {
            is CaldavTestStatus.Success -> {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = stringResource(Res.string.settings_caldav_test_success),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
            is CaldavTestStatus.Error -> {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = testStatus.message,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
            is CaldavTestStatus.Testing -> {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = stringResource(Res.string.settings_caldav_testing),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            CaldavTestStatus.Idle -> Unit
        }
    }
}
