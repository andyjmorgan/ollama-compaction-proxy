package tokencount

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCountRequiresValidNonnegativeTokens(t *testing.T) {
	for _, body := range []string{`{}`, `{"tokens":-1}`, `{"tokens":"100"}`, `null`} {
		t.Run(body, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(body)) }))
			defer srv.Close()
			if _, err := New(srv.URL).CountAnthropic(context.Background(), []byte(`{"model":"m"}`)); err == nil {
				t.Fatal("invalid count accepted")
			}
		})
	}
}
