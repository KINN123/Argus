package registry

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAISBenchRunnersRegisterAndValidateExamples(t *testing.T) {
	for _, name := range []string{"aisbench-acc", "aisbench-perf"} {
		t.Run(name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("..", "..", "..", "runners", name, "runner.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			spec, err := Parse(raw)
			if err != nil {
				t.Fatal(err)
			}
			if spec.Name != name || spec.Version != "0.2.0" {
				t.Fatalf("unexpected runner identity: %s %s", spec.Name, spec.Version)
			}
			if issues := ValidateInput(spec, spec.Placeholder); len(issues) != 0 {
				t.Fatalf("invalid placeholder: %+v", issues)
			}
			if name == "aisbench-perf" {
				input := `{"metrics_endpoint":["http://127.0.0.1:7800/metrics"],"rounds":[{"input_len":256,"seed":1},{"input_len":512,"seed":2}],"repeat_rate":"50%"}`
				if issues := ValidateInput(spec, input); len(issues) != 0 {
					t.Fatalf("invalid multi-round input: %+v", issues)
				}
			}
		})
	}
}
